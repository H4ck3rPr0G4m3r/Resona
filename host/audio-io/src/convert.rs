//! Interleaved device bytes ↔ mono f32.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleFormat {
    F32,
    I16,
    I24In32, // 24 valid bits left-justified in 32-bit containers
    I32,
}

impl SampleFormat {
    pub fn from_wave(fmt: &wasapi::WaveFormat) -> Option<Self> {
        match (fmt.get_subformat().ok()?, fmt.get_bitspersample(), fmt.get_validbitspersample()) {
            (wasapi::SampleType::Float, 32, _) => Some(Self::F32),
            (wasapi::SampleType::Int, 16, _) => Some(Self::I16),
            (wasapi::SampleType::Int, 32, 24) => Some(Self::I24In32),
            (wasapi::SampleType::Int, 32, _) => Some(Self::I32),
            _ => None,
        }
    }

    pub fn bytes(self) -> usize {
        match self {
            Self::I16 => 2,
            _ => 4,
        }
    }

    #[inline]
    fn read(self, b: &[u8]) -> f32 {
        match self {
            Self::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            Self::I16 => i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0,
            Self::I24In32 | Self::I32 => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32 / 2_147_483_648.0,
        }
    }

    #[inline]
    fn write(self, v: f32, b: &mut [u8]) {
        let v = v.clamp(-1.0, 1.0);
        match self {
            Self::F32 => b[..4].copy_from_slice(&v.to_le_bytes()),
            Self::I16 => b[..2].copy_from_slice(&((v * 32767.0).round() as i16).to_le_bytes()),
            Self::I24In32 | Self::I32 => b[..4].copy_from_slice(&((v as f64 * 2_147_483_647.0).round() as i32).to_le_bytes()),
        }
    }
}

/// Average all channels of interleaved `src` into `dst` (one sample per frame).
pub fn to_mono(src: &[u8], fmt: SampleFormat, channels: usize, dst: &mut [f32]) -> usize {
    let frame_bytes = fmt.bytes() * channels;
    let frames = (src.len() / frame_bytes).min(dst.len());
    let inv = 1.0 / channels as f32;
    for (f, out) in dst.iter_mut().enumerate().take(frames) {
        let base = f * frame_bytes;
        let mut acc = 0.0;
        for c in 0..channels {
            acc += fmt.read(&src[base + c * fmt.bytes()..]);
        }
        *out = acc * inv;
    }
    frames
}

/// Copy mono `src` to every channel of interleaved `dst`.
pub fn from_mono(src: &[f32], fmt: SampleFormat, channels: usize, dst: &mut [u8]) {
    let frame_bytes = fmt.bytes() * channels;
    for (f, &v) in src.iter().enumerate() {
        let base = f * frame_bytes;
        for c in 0..channels {
            fmt.write(v, &mut dst[base + c * fmt.bytes()..]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_f32_stereo() {
        let src = [0.25f32, -0.5, 1.0];
        let mut bytes = vec![0u8; 3 * 2 * 4];
        from_mono(&src, SampleFormat::F32, 2, &mut bytes);
        let mut back = [0f32; 3];
        assert_eq!(to_mono(&bytes, SampleFormat::F32, 2, &mut back), 3);
        assert_eq!(back, src);
    }

    #[test]
    fn roundtrip_i16() {
        let src = [0.5f32, -0.5];
        let mut bytes = vec![0u8; 2 * 2];
        from_mono(&src, SampleFormat::I16, 1, &mut bytes);
        let mut back = [0f32; 2];
        to_mono(&bytes, SampleFormat::I16, 1, &mut back);
        assert!((back[0] - 0.5).abs() < 1e-4 && (back[1] + 0.5).abs() < 1e-4);
    }
}
