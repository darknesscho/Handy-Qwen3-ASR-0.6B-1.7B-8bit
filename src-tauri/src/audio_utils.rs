//! Audio utility functions for format conversion
//! 
//! This module provides utilities for converting audio data between
//! different formats, such as PCM float32 to WAV.

use log::debug;

/// Convert PCM float32 audio data to WAV format
/// 
/// # Arguments
/// * `pcm_data` - PCM audio samples as f32 array (normalized -1.0 to 1.0)
/// * `sample_rate` - Sample rate in Hz (e.g., 16000)
/// 
/// # Returns
/// * `Vec<u8>` - WAV file data
pub fn pcm_to_wav(pcm_data: &[f32], sample_rate: u32) -> Vec<u8> {
    debug!("Converting {} PCM samples to WAV format", pcm_data.len());
    
    let num_channels: u16 = 1; // Mono
    let bits_per_sample: u16 = 16;
    let byte_rate: u32 = sample_rate * num_channels as u32 * (bits_per_sample / 8) as u32;
    let block_align: u16 = num_channels * bits_per_sample / 8;
    let data_size: u32 = (pcm_data.len() as u32) * (bits_per_sample / 8) as u32;
    let file_size: u32 = 36 + data_size;

    let mut wav_data = Vec::with_capacity(44 + pcm_data.len() * 2);

    // RIFF header
    wav_data.extend_from_slice(b"RIFF");
    wav_data.extend_from_slice(&file_size.to_le_bytes());
    wav_data.extend_from_slice(b"WAVE");

    // fmt subchunk
    wav_data.extend_from_slice(b"fmt ");
    wav_data.extend_from_slice(&16u32.to_le_bytes()); // Subchunk1Size (16 for PCM)
    wav_data.extend_from_slice(&1u16.to_le_bytes()); // AudioFormat (1 for PCM)
    wav_data.extend_from_slice(&num_channels.to_le_bytes());
    wav_data.extend_from_slice(&sample_rate.to_le_bytes());
    wav_data.extend_from_slice(&byte_rate.to_le_bytes());
    wav_data.extend_from_slice(&block_align.to_le_bytes());
    wav_data.extend_from_slice(&bits_per_sample.to_le_bytes());

    // data subchunk
    wav_data.extend_from_slice(b"data");
    wav_data.extend_from_slice(&data_size.to_le_bytes());

    // Convert f32 samples to i16
    for &sample in pcm_data {
        // Clamp to [-1.0, 1.0] range
        let clamped = sample.max(-1.0).min(1.0);
        // Convert to i16
        let int_sample = (clamped * i16::MAX as f32) as i16;
        wav_data.extend_from_slice(&int_sample.to_le_bytes());
    }

    debug!("Generated WAV file: {} bytes", wav_data.len());
    
    wav_data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pcm_to_wav_conversion() {
        // Create a simple sine wave
        let sample_rate = 16000;
        let duration = 0.1; // 100ms
        let frequency = 440.0; // A4
        
        let samples: Vec<f32> = (0..(sample_rate as f32 * duration))
            .map(|t| (2.0 * std::f32::consts::PI * frequency * t / sample_rate as f32).sin())
            .collect();
        
        let wav_data = pcm_to_wav(&samples, sample_rate);
        
        // Check WAV header
        assert_eq!(&wav_data[0..4], b"RIFF");
        assert_eq!(&wav_data[8..12], b"WAVE");
        assert_eq!(&wav_data[12..16], b"fmt ");
        assert_eq!(&wav_data[36..40], b"data");
        
        // Check file size (44 byte header + 2 bytes per sample)
        let expected_size = 44 + samples.len() * 2;
        assert_eq!(wav_data.len(), expected_size);
    }

    #[test]
    fn test_empty_audio() {
        let wav_data = pcm_to_wav(&[], 16000);
        assert_eq!(wav_data.len(), 44); // Header only
    }
}
