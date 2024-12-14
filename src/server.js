import express from 'express';
import cors from 'cors';
import multer from 'multer';
import path from 'path';
import { fileURLToPath } from 'url';
import { processAudio } from './services/audioProcessor.mjs';
import { transcribeAudio } from './services/transcriptionService.mjs';
import { qaService } from './services/qaService.mjs';
import dotenv from 'dotenv';

dotenv.config();

// Get __dirname equivalent in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const upload = multer({ 
  storage: multer.diskStorage({
    destination: 'uploads/',
    filename: (req, file, cb) => {
      const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
      cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
    }
  }),
  fileFilter: (req, file, cb) => {
    const allowedTypes = ['audio/mpeg', 'audio/wav', 'video/mp4'];
    if (allowedTypes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('Invalid file type. Only audio and video files are allowed.'));
    }
  },
  limits: {
    fileSize: 50 * 1024 * 1024 // 50MB limit
  }
});

// Middleware
app.use(cors());
app.use(express.json());
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// Routes
app.get('/', (req, res) => {
  res.send('Hello from the server!');
});

app.post('/api/upload', upload.single('media'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    const file = req.file;
    let audioPath = file.path;

    // If it's a video file, extract audio
    if (file.mimetype === 'video/mp4') {
      audioPath = await processAudio(file.path);
    }

    console.log('audioPath:', audioPath);

    // Transcribe audio
    const transcription = await transcribeAudio(audioPath);

    res.json({
      id: path.basename(file.path),
      originalName: file.originalname,
      transcript: transcription
    });

  } catch (error) {
    console.error('Upload error:', error);
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/ask', async (req, res) => {
  try {
    console.log('Received question:', req.body.question);
    const { mediaId, question } = req.body;

    if (!mediaId || !question) {
      return res.status(400).json({ error: 'Missing required parameters' });
    }

    const answer = await qaService.answerQuestion(mediaId, question);
    res.json(answer);

  } catch (error) {
    console.error('Question answering error:', error);
    res.status(500).json({ error: error.message });
  }
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});