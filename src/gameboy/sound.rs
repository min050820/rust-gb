use sdl2::{AudioSubsystem, Sdl, EventPump};
use std::rc::Rc;
use std::sync::Mutex;
use sdl2::audio::{AudioSpec, AudioFormat, AudioSpecDesired, AudioCallback, AudioDevice, AudioCVT};
use std::sync::mpsc::{SyncSender, Receiver, RecvTimeoutError};
use std::io::Write;
use std::time::Duration;

const MASTER_CLOCK: i32 = 4194304;
const FREQ_DIVISOR: i32 = 4;
const MASTER_FREQ: i32 = MASTER_CLOCK / FREQ_DIVISOR;

struct Callback {
    audio_rx: Receiver<[[f32; 2]; 256]>,
}

impl AudioCallback for Callback {
    type Channel = f32;

    fn callback(&mut self, buf: &mut [f32]) {
        match self.audio_rx.recv_timeout(Duration::from_millis(30)) {
            Ok(x) => {
                for i in 0..x.len() {
                    buf[i * 2] = x[i][0];
                    buf[i * 2 + 1] = x[i][1];
                }
            },
            Err(_) => {
                for i in 0..buf.len() {
                    buf[i] = 0.;
                }
            },
        }

        for i in 0..buf.len() {
            buf[i] *= 0.25;
        }
    }
}

struct AudioProcessor {
    high_pass_average: f32,
    input_buffer_pos: usize,
    output_buffer_pos: usize,
    output_buffer_cnt: i32,
    input_buffer: [[f32; 2]; 1024],
    output_buffer: [[f32; 2]; 1024],
}

// Can only process [high freq] -> [low freq]
impl AudioProcessor {
    fn new() -> AudioProcessor {
        AudioProcessor {
            high_pass_average: 0.,
            input_buffer_pos: 0,
            output_buffer_pos: 0,
            output_buffer_cnt: 0,
            input_buffer: [[0.; 2]; 1024],
            output_buffer: [[0.; 2]; 1024],
        }
    }

    fn push(&mut self, sample_l: f32, sample_r: f32) {
        self.input_buffer[self.input_buffer_pos][0] = sample_l;
        self.input_buffer[self.input_buffer_pos][1] = sample_r;
        self.input_buffer_pos += 1;
        if self.input_buffer_pos == self.input_buffer.len() {
            self.process();
            self.input_buffer_pos = 0;
        }
    }

    fn high_pass(&mut self) {
        let mut sum = 0.0;
        for i in 0..self.input_buffer.len() {
            sum += self.input_buffer[i][0];
            sum += self.input_buffer[i][1];
        }
        sum /= (self.input_buffer.len() * 2) as f32;
        self.high_pass_average = (self.high_pass_average * 0.98) + (sum * 0.02);
        for i in 0..self.input_buffer.len() {
            self.input_buffer[i][0] -= self.high_pass_average;
            self.input_buffer[i][1] -= self.high_pass_average;
        }
    }

    fn process(&mut self) {
        self.high_pass();
        let pack_ratio = f64::round(MASTER_FREQ as f64 / 48000.0) as i32;
        for i in 0..self.input_buffer.len() {
            self.output_buffer[self.output_buffer_pos][0] += self.input_buffer[i][0];
            self.output_buffer[self.output_buffer_pos][1] += self.input_buffer[i][1];
            self.output_buffer_cnt += 1;
            if self.output_buffer_cnt >= pack_ratio {
                self.output_buffer[self.output_buffer_pos][0] /= pack_ratio as f32;
                self.output_buffer[self.output_buffer_pos][1] /= pack_ratio as f32;
                self.output_buffer_cnt = 0;
                self.output_buffer_pos += 1;
                self.output_buffer[self.output_buffer_pos][0] = 0.;
                self.output_buffer[self.output_buffer_pos][1] = 0.;
            }
        }
    }

    fn pop(&mut self) -> Option<[[f32; 2]; 256]> {
        // Last sample is not yet complete; Don't use that sample
        if self.output_buffer_pos <= 256 {
            None
        }
        else {
            let mut ret = [[0.; 2]; 256];
            ret.copy_from_slice(&self.output_buffer[0..256]);
            for i in 256..self.output_buffer_pos {
                self.output_buffer[i-256] = self.output_buffer[i];
            }
            self.output_buffer_pos -= 256;
            Some(ret)
        }
    }
}

#[derive(Copy, Clone)]
struct Clock {
    curr_tick: i32,
}

impl Clock {
    fn new() -> Clock {
        Clock { curr_tick: 0 }
    }

    fn tick(&mut self) {
        self.curr_tick += 1;
        if self.curr_tick >= MASTER_FREQ {
            self.curr_tick = 0;
        }
    }

    fn derive_duration(&self, dur: i32) -> bool {
        assert_eq!(MASTER_FREQ % dur, 0);
        self.curr_tick % dur == 0
    }

    fn derive_freq(&self, freq: i32) -> bool {
        self.derive_duration(MASTER_FREQ / freq)
    }
}

#[derive(Copy, Clone)]
struct FrequencySweep {
    freq: u16,
    inc: bool,
    shift_amount: u8,
    timer_await: i32,
    update_rate: i32,
}

impl FrequencySweep {
    fn new(nrx0: u8) -> FrequencySweep {
        let update_rate = (nrx0 & 0x70) >> 4;
        let inc = (nrx0 & 8) == 0;
        let shift_amount = nrx0 & 3;
        FrequencySweep {
            freq: 0,
            inc,
            shift_amount,
            timer_await: 0,
            update_rate: update_rate as i32
        }
    }

    fn default() -> FrequencySweep {
        Self::new(0x00)
    }

    fn is_enabled(&self) -> bool {
        self.update_rate != 0
    }

    fn tick(&mut self, clock: &Clock) -> Option<u16> {
        assert_ne!(self.update_rate, 0);

        if clock.derive_freq(128) {
            self.timer_await += 1;
            if self.update_rate <= self.timer_await {
                self.timer_await = 0;
                let next = self.next_freq(self.freq);
                let predict = self.next_freq(next);
                if predict <= 0x7FF {
                    self.freq = next;
                } else {
                    return None;
                }
            }
        }

        Some(self.freq)
    }

    fn next_freq(&self, freq: u16) -> u16 {
        let modifier = freq >> (self.shift_amount + 4);
        if self.inc {
            freq.saturating_add(modifier)
        } else {
            freq.wrapping_add(!modifier)
        }
    }

    fn reset(&mut self, freq: u16) {
        self.freq = freq;
        self.timer_await = 0;
    }

    fn reload(&mut self, nrx0: u8) {
        let prev_freq = self.freq;
        *self = FrequencySweep::new(nrx0);
        self.freq = prev_freq;
    }
}

#[derive(Copy, Clone)]
struct VolumeSweep {
    curr_vol: i8,
    change_amount: i8,
    sweep_time: i32,
    sweep_await: i32,
}

impl VolumeSweep {
    fn new(nrx2: u8) -> VolumeSweep {
        let inc = nrx2 & 8 != 0;
        let sweep_time = (nrx2 & 7) as i8;
        VolumeSweep {
            curr_vol: ((nrx2 >> 4) & 0xF) as i8,
            change_amount: if inc {1} else {-1},
            sweep_time: sweep_time as i32,
            sweep_await: 0,
        }
    }

    fn default() -> VolumeSweep {
        Self::new(0x00)
    }

    fn tick(&mut self, clock: &Clock) -> f32 {
        if self.sweep_time != 0 && clock.derive_freq(64) {
            self.sweep_await -= 1;
            if self.sweep_await <= 0 {
                let new_vol = self.curr_vol + self.change_amount;
                self.sweep_await = self.sweep_time;
                if 0 <= new_vol && new_vol <= 15 {
                    self.curr_vol = new_vol;
                }
            }
        }
        (self.curr_vol as f32) / 15.
    }
}

#[derive(Copy, Clone)]
struct LengthTimer {
    enabled: bool,
    length: i32,
}

impl LengthTimer {
    fn tick(&mut self, clock: &Clock) -> bool {
        if !self.enabled {
            true
        } else if self.length <= 0 {
            false
        } else {
            if clock.derive_freq(256) {
                self.length -= 1;
            }
            true
        }
    }

    fn set_length(&mut self, len: i32) {
        self.length = len;
    }
}

#[derive(Copy, Clone)]
struct Tone {
    freq_sweep: FrequencySweep,
    volume_sweep: VolumeSweep,
    length_timer: LengthTimer,
    has_freq_sweep: bool,
    written_samples: i32,
    nrx0: u8,
    nrx1: u8,
    nrx2: u8,
    nrx3: u8,
    nrx4: u8,
}

impl Tone {
    fn new(has_freq_sweep: bool) -> Tone {
        Tone {
            freq_sweep: FrequencySweep::default(),
            volume_sweep: VolumeSweep::default(),
            length_timer: LengthTimer { enabled: false, length: 0 },
            has_freq_sweep,
            written_samples: 0,
            nrx0: 0,
            nrx1: 0,
            nrx2: 0,
            nrx3: 0,
            nrx4: 0,
        }
    }

    fn get_freq_val(&self) -> u16 {
        (self.nrx3 as u16) | (((self.nrx4 & 0b111) as u16) << 8)
    }

    fn get_period(&self, freq_val: u16) -> i32 {
        let period = (2048 - freq_val as i32) * (MASTER_FREQ / 131072);
        period as i32
    }

    fn get_duty(&self) -> f32 {
        let duty = ((self.nrx1 >> 6) & 0x03) as usize;
        [0.125, 0.25, 0.5, 0.75][duty]
    }

    fn tick(&mut self, clock: &Clock) -> f32 {
        if !self.length_timer.tick(clock) {
            return 0.;
        }

        let freq_val = if self.freq_sweep.is_enabled() {
            match self.freq_sweep.tick(clock) {
                Some(x) => x,
                None => { return 0.; }
            }
        } else {
            self.get_freq_val()
        };

        let vol = self.volume_sweep.tick(clock);

        let period = self.get_period(freq_val);
        if self.written_samples >= period {
            self.written_samples = 0;
        }
        let period_progress = self.written_samples as f32 / (period - 1) as f32;
        self.written_samples += 1;
        if period_progress < self.get_duty() {
            vol
        } else {
            -vol
        }
    }

    fn write_reg(&mut self, reg: u8, data: u8) {
        match reg {
            0 => {
                self.nrx0 = data;
                if self.has_freq_sweep {
                    self.freq_sweep.reload(data);
                }
            },
            1 => {
                self.nrx1 = data;
                self.length_timer.set_length((data & 0x3F) as i32);
            },
            2 => {
                self.nrx2 = data;
                self.volume_sweep = VolumeSweep::new(data);
            },
            3 => self.nrx3 = data,
            4 => {
                self.nrx4 = data;
                if data & 0x80 != 0 {
                    self.written_samples = 0;
                    if self.has_freq_sweep {
                        self.freq_sweep.reset(self.get_freq_val());
                    }
                }
                self.length_timer.enabled = (data & 0x40 != 0);
            },
            _ => unreachable!()
        }
    }
}

#[derive(Copy, Clone)]
struct Wave {
    length_timer: LengthTimer,
    written_samples: i32,
    pos: usize,
    nrx0: u8,
    nrx1: u8,
    nrx2: u8,
    nrx3: u8,
    nrx4: u8,
    wave_table: [u8; 32]
}

impl Wave {
    fn new() -> Wave {
        Wave {
            length_timer: LengthTimer { enabled: false, length: 0 },
            written_samples: 0,
            pos: 0,
            nrx0: 0,
            nrx1: 0,
            nrx2: 0,
            nrx3: 0,
            nrx4: 0,
            wave_table: [0xFF; 32]
        }
    }

    fn tick(&mut self, clock: &Clock) -> f32 {
        if self.nrx0 & 0x80 == 0 || !self.length_timer.tick(clock) {
            return 0.;
        }

        let freq_val = self.nrx3 as u16 | ((self.nrx4 as u16 & 7) << 8);
        let period = 2 * (2048 - freq_val as i32);
        if self.written_samples >= period {
            self.written_samples = 0;
            self.pos += 1;
            if self.pos == 32 {
                self.pos = 0;
            }
        }

        self.written_samples += FREQ_DIVISOR;
        let vol = [0.0, 1.0, 0.5, 0.25][((self.nrx2 >> 5) & 3) as usize];
        let sample = self.wave_table[self.pos] as f32 / 15.0;
        (sample * 2.0 - 1.0) * vol
    }

    fn write_reg(&mut self, reg: u8, data: u8) {
        match reg {
            0 => {
                self.nrx0 = data;
            },
            1 => {
                self.nrx1 = data;
                self.length_timer.set_length(data as i32);
            },
            2 => self.nrx2 = data,
            3 => self.nrx3 = data,
            4 => {
                self.nrx4 = data;
                if data & 0x80 != 0 {
                    //self.written_samples = 0;
                }
                self.length_timer.enabled = (data & 0x40 != 0);
            },
            _ => unreachable!()
        }
    }

    fn write_table(&mut self, addr: usize, data: u8) {
        self.wave_table[addr * 2] = (data & 0xF0) >> 4;
        self.wave_table[addr * 2 + 1] = data & 0x0F;
    }

    fn read_table(&self, addr: usize) -> u8 {
        (self.wave_table[addr * 2] << 4) | self.wave_table[addr * 2 + 1]
    }
}

#[derive(Copy, Clone)]
struct Noise {
    length_timer: LengthTimer,
    volume_sweep: VolumeSweep,
    written_samples: i32,
    noise_sample: u16,
    nrx1: u8,
    nrx2: u8,
    nrx3: u8,
    nrx4: u8,
}

impl Noise {
    fn new() -> Noise {
        Noise {
            length_timer: LengthTimer { enabled: false, length: 0 },
            volume_sweep: VolumeSweep::default(),
            written_samples: 0,
            noise_sample: 0xFFFF,
            nrx1: 0,
            nrx2: 0,
            nrx3: 0,
            nrx4: 0
        }
    }

    fn get_period(&self) -> i32 {
        let shift = (self.nrx3 & 0xF0) >> 4;
        let divisor = (self.nrx3 & 7) as usize;
        let divisor = [8, 16, 32, 48, 64, 80, 96, 112][divisor];
        divisor << (shift + 1)
    }

    fn tick_sample(&mut self) {
        let mut carry = self.noise_sample & 1 != 0;
        let mut value = self.noise_sample >> 1;
        carry ^= (value & 1 != 0);
        value &= 0x3FFF;
        if carry {
            value |= 0x4000;
        }
        if self.nrx3 & 8 != 0 {
            value &= 0x7F;
            if carry {
                value |= 0x80;
            }
        }
        self.noise_sample = value;
    }

    fn tick(&mut self, clock: &Clock) -> f32 {
        if !self.length_timer.tick(clock) {
            return 0.;
        }
        let vol = self.volume_sweep.tick(clock);
        let period = self.get_period();

        self.written_samples += FREQ_DIVISOR;
        if period as i32 <= self.written_samples {
            self.written_samples = 0;
            self.tick_sample();
        }

        if self.noise_sample & 1 != 0 {
            vol
        } else {
            -vol
        }
    }

    fn write_reg(&mut self, reg: u8, data: u8) {
        match reg {
            1 => {
                self.nrx1 = data;
                self.length_timer.set_length((data & 0x3F) as i32);
            },
            2 => {
                self.nrx2 = data;
                self.volume_sweep = VolumeSweep::new(data);
            },
            3 => self.nrx3 = data,
            4 => {
                self.nrx4 = data;
                self.length_timer.enabled = (data & 0x40 != 0);
                if data & 0x80 != 0 {
                    self.written_samples = 0;
                    self.noise_sample = 0xFFFF;
                }
            },
            _ => unreachable!()
        }
    }
}

pub struct Sound {
    clock: Clock,
    tone1: Tone,
    tone2: Tone,
    wave: Wave,
    noise: Noise,
    nr50: u8,
    nr51: u8,
    //event_pump: Rc<Mutex<EventPump>>,
    audio_cvt: AudioProcessor,
    audio: AudioSubsystem,
    player: AudioDevice<Callback>,
    audio_tx: SyncSender<[[f32; 2]; 256]>,
}

impl Sound {
    pub fn new(sdl: &Sdl) -> Sound {
        let (audio_tx, audio_rx) = std::sync::mpsc::sync_channel(4);
        let audio = sdl.audio().unwrap();
        let spec = AudioSpecDesired {
            freq: Some(48000),
            channels: Some(2),
            samples: Some(256)
        };
        let player = audio.open_playback(None, &spec, |spec| {
            println!("Open audio device: {:?}", spec);
            Callback {audio_rx}
        }).unwrap();
        player.resume();
        let audio_cvt = AudioProcessor::new();
        Sound {
            clock: Clock::new(),
            tone1: Tone::new(true),
            tone2: Tone::new(false),
            wave: Wave::new(),
            noise: Noise::new(),
            nr50: 0,
            nr51: 0,
            audio_cvt,
            audio,
            player,
            audio_tx,
        }
    }

    fn write_sample(&mut self, sample_l: f32, sample_r: f32) {
        self.audio_cvt.push(sample_l, sample_r);
        match self.audio_cvt.pop() {
            Some(x) => {
                self.audio_tx.send(x).unwrap();
            },
            None => {}
        }
    }

    pub fn read(&mut self, addr: u16) -> u8 {
        match addr {
            0xFF11 => self.tone1.nrx1,
            0xFF16 => self.tone2.nrx1,
            0xFF24 => self.nr50,
            0xFF25 => self.nr51,
            0xFF30..=0xFF3F => {
                let pos = (addr - 0xFF30) as usize;
                self.wave.read_table(pos)
            }
            x => {
                println!("Unknown sound read: {:04x}", addr);
                0xFF
            }
        }
    }

    pub fn write(&mut self, addr: u16, data: u8) {
        match addr {
            0xFF10 => self.tone1.write_reg(0, data),
            0xFF11 => self.tone1.write_reg(1, data),
            0xFF12 => self.tone1.write_reg(2, data),
            0xFF13 => self.tone1.write_reg(3, data),
            0xFF14 => self.tone1.write_reg(4, data),
            0xFF16 => self.tone2.write_reg(1, data),
            0xFF17 => self.tone2.write_reg(2, data),
            0xFF18 => self.tone2.write_reg(3, data),
            0xFF19 => self.tone2.write_reg(4, data),
            0xFF1A => self.wave.write_reg(0, data),
            0xFF1B => self.wave.write_reg(1, data),
            0xFF1C => self.wave.write_reg(2, data),
            0xFF1D => self.wave.write_reg(3, data),
            0xFF1E => self.wave.write_reg(4, data),
            0xFF20 => self.noise.write_reg(1, data),
            0xFF21 => self.noise.write_reg(2, data),
            0xFF22 => self.noise.write_reg(3, data),
            0xFF23 => self.noise.write_reg(4, data),
            0xFF24 => self.nr50 = data,
            0xFF25 => self.nr51 = data,
            0xFF30..=0xFF3F => {
                let pos = (addr - 0xFF30) as usize;
                self.wave.write_table(pos, data);
            }
            x => {
                //println!("Unknown sound write: {:04x}={:02x}", addr, data);
            }
        }
    }

    pub fn tick(&mut self, cycles: i32) -> u8 {
        for _ in 0..cycles/FREQ_DIVISOR {
            let mut samples = [0.0; 4];
            samples[0] = self.tone1.tick(&self.clock);
            samples[1] = self.tone2.tick(&self.clock);
            samples[2] = self.wave.tick(&self.clock);
            samples[3] = self.noise.tick(&self.clock);

            let mut filter = self.nr51;
            let mut sample_r = 0.0;
            for i in 0..4 {
                if filter & 1 != 0 {
                    sample_r += samples[i];
                }
                filter >>= 1;
            }
            sample_r *= 0.25 * ((self.nr50 & 7) as f32 / 7.);

            let mut sample_l = 0.0;
            for i in 0..4 {
                if filter & 1 != 0 {
                    sample_l += samples[i];
                }
                filter >>= 1;
            }
            sample_l *= 0.25 * (((self.nr50 >> 4) & 7) as f32 / 7.);

            self.write_sample(sample_l, sample_r);
            self.clock.tick();
        }

        0
    }
}