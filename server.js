const express = require('express');
const multer = require('multer');
const tf = require('@tensorflow/tfjs-node');
const { v4: uuidv4 } = require('uuid');
const cors = require('cors');
const fs = require('fs'); // Untuk menyimpan prediksi lokal

// Konfigurasi penyimpanan file sementara
const upload = multer({
  limits: { fileSize: 1000000 }, // Maksimal 1MB
  fileFilter: (req, file, cb) => {
    const allowedTypes = ['image/jpeg', 'image/png'];
    if (allowedTypes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('Invalid file type. Only JPEG and PNG are allowed.'));
    }
  },
}).single('image'); // Pastikan field name yang digunakan di form-data adalah 'image'

const app = express();
app.use(cors()); // Menambahkan middleware CORS agar frontend dapat mengakses API
app.use(express.json());

// Route GET untuk root
app.get('/', (req, res) => {
  res.status(200).json({
    message: 'Welcome to the Cancer Prediction API!',
  });
});

// Muat model TensorFlow dari path lokal
let model;
async function loadModel() {
  try {
    console.log('Loading model...');
    model = await tf.loadGraphModel('../ml/model.json'); // Path lokal ke model
    console.log('Model loaded successfully');
  } catch (error) {
    console.error('Error loading model:', error);
    process.exit(1); // Keluar jika model gagal dimuat
  }
}

// Fungsi untuk melakukan prediksi
async function predictClassification(model, image) {
  try {
    const tensor = tf.node
      .decodeJpeg(image)
      .resizeNearestNeighbor([224, 224])
      .expandDims()
      .toFloat();

    const prediction = model.predict(tensor);
    const score = await prediction.data();
    const confidenceScore = Math.max(...score) * 100;

    const isCancer = confidenceScore > 50;
    const label = isCancer ? 'Cancer' : 'Non-cancer';
    const suggestion = isCancer
      ? "Segera periksa ke dokter!"
      : "Penyakit kanker tidak terdeteksi.";

    return { label, suggestion };
  } catch (error) {
    throw new Error(`Terjadi kesalahan input: ${error.message}`);
  }
}

// Endpoint POST untuk prediksi
app.post('/predict', async (req, res) => {
  upload(req, res, async (err) => {
    if (err instanceof multer.MulterError) {
      console.error('Multer error:', err);
      return res.status(413).json({
        status: 'fail',
        message: 'Payload content length greater than maximum allowed: 1000000',
      });
    } else if (err) {
      console.error('Error:', err);
      return res.status(400).json({
        status: 'fail',
        message: err.message,
      });
    }

    if (!req.file) {
      console.error('No file uploaded');
      return res.status(400).json({
        status: 'fail',
        message: 'No file uploaded',
      });
    }

    const imageBuffer = req.file.buffer;

    try {
      const { label, suggestion } = await predictClassification(model, imageBuffer);

      const predictionDoc = {
        id: uuidv4(),
        result: label,
        suggestion,
        createdAt: new Date().toISOString(),
      };

      // Simpan ke file lokal
      const predictions = JSON.parse(fs.readFileSync('predictions.json', 'utf-8') || '[]');
      predictions.push(predictionDoc);
      fs.writeFileSync('predictions.json', JSON.stringify(predictions, null, 2));

      console.log('Prediction saved locally:', predictionDoc);

      res.status(201).json({
        status: 'success',
        message: 'Model is predicted successfully',
        data: predictionDoc,
      });
    } catch (error) {
      console.error('Prediction error:', error);
      res.status(400).json({
        status: 'fail',
        message: 'Terjadi kesalahan dalam melakukan prediksi',
      });
    }
  });
});

// Middleware untuk menangani error
app.use((err, req, res, next) => {
  if (err.code === 'LIMIT_FILE_SIZE') {
    return res.status(413).json({
      status: 'fail',
      message: 'Payload content length greater than maximum allowed: 1000000',
    });
  }
  console.error('Unhandled error:', err);
  res.status(500).json({
    status: 'fail',
    message: 'Internal server error',
  });
});

// Mulai server
const PORT = process.env.PORT || 3000;
app.listen(PORT, async () => {
  await loadModel();
  console.log(`Server is running on port ${PORT}`);
});
