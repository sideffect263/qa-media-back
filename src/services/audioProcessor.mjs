// src/services/audioProcessor.mjs
import ffmpeg from 'fluent-ffmpeg';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export const processAudio = async (videoPath) => {
  const audioPath = videoPath.replace(/\.[^/.]+$/, '.mp3');
  
  return new Promise((resolve, reject) => {
    ffmpeg(videoPath)
      .toFormat('mp3')
      .on('end', () => {
        console.log('Audio extraction completed:', audioPath);
        resolve(audioPath);
      })
      .on('error', (err) => {
        console.error('Audio extraction error:', err);
        reject(new Error(`Audio extraction failed: ${err.message}`));
      })
      .on('progress', (progress) => {
        console.log('Processing: ' + progress.percent + '% done');
      })
      .save(audioPath);
  });
};

export default {
  processAudio
};