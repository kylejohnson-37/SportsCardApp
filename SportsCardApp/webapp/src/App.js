import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import * as tf from '@tensorflow/tfjs';
import Papa from 'papaparse';
import { 
  Box, 
  Button, 
  Container, 
  Typography, 
  Paper, 
  AppBar, 
  Toolbar, 
  IconButton,
  CircularProgress
} from '@mui/material';
import { createTheme, ThemeProvider } from '@mui/material/styles';

// Create a theme
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <SportsCardDetector />
    </ThemeProvider>
  );
}

function SportsCardDetector() {
  const [model, setModel] = useState(null);
  const [cardData, setCardData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Load model and card data on component mount
  useEffect(() => {
    const loadModel = async () => {
      setLoading(true);
      try {
        // Create a simple placeholder model
        // This will be replaced with your actual TensorFlow model later
        const simpleModel = tf.sequential();
        simpleModel.add(tf.layers.conv2d({
          inputShape: [224, 224, 3],
          kernelSize: 3,
          filters: 16,
          activation: 'relu'
        }));
        simpleModel.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));
        simpleModel.add(tf.layers.flatten());
        simpleModel.add(tf.layers.dense({units: 10, activation: 'softmax'}));
        
        // Compile the model
        simpleModel.compile({
          optimizer: 'adam',
          loss: 'categoricalCrossentropy',
          metrics: ['accuracy']
        });
        
        setModel(simpleModel);
        console.log('Placeholder model created successfully');
      } catch (error) {
        console.error('Error creating model:', error);
        setError(`Failed to create the model. Error: ${error.message}`);
      }
      setLoading(false);
    };

    // Load the card data
    const loadCardData = async () => {
      try {
        const response = await fetch('/cards.csv');
        const csvText = await response.text();
        const parsedData = Papa.parse(csvText, { header: true }).data;
        setCardData(parsedData);
        console.log('Card data loaded:', parsedData.length, 'cards');
      } catch (error) {
        console.error('Error loading card data:', error);
      }
    };

    loadModel();
    loadCardData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Sports Card Detector
          </Typography>
        </Toolbar>
      </AppBar>
      <Container maxWidth="sm" sx={{ py: 6 }}>
        <Typography variant="h4" align="center" gutterBottom>
          Sports Card Detection
        </Typography>
        <Typography variant="body1" align="center" color="text.secondary" gutterBottom>
          Share your camera feed and let our AI detect sports cards in real time!
        </Typography>
        <CameraFeedWithOverlay 
          model={model} 
          cardData={cardData} 
          loading={loading} 
          error={error} 
        />
      </Container>
    </>
  );
}

function CameraFeedWithOverlay({ model, cardData, loading, error: appError }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [error, setError] = useState(appError);
  const [prediction, setPrediction] = useState(null);

  // Set up video stream when camera is activated
  useEffect(() => {
    let stream;
    let predictionInterval;
    // Store a reference to the current video element that will be used in cleanup
    const videoElement = videoRef.current;
    
    if (cameraActive) {
      setError(null);
      const setupCamera = async () => {
        try {
          stream = await navigator.mediaDevices.getUserMedia({ video: true });
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            videoRef.current.onloadedmetadata = () => {
              // Start prediction loop
              predictionInterval = setInterval(() => {
                detectFrame();
              }, 100); // Run every 100ms
            };
          }
        } catch (err) {
          console.error('Error accessing camera:', err);
          setError('Could not access camera. Please ensure camera permissions are granted.');
        }
      };
      setupCamera();
    }
    
    return () => {
      if (predictionInterval) {
        clearInterval(predictionInterval);
      }
      // Use the stored reference in cleanup
      if (videoElement && videoElement.srcObject) {
        videoElement.srcObject.getTracks().forEach(track => track.stop());
        videoElement.srcObject = null;
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cameraActive]);

  const detectFrame = React.useCallback(async () => {
    if (model && videoRef.current && videoRef.current.readyState === 4) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');

      // Set canvas dimensions to match video
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      let topPrediction = null;
      
      try {
        // Pre-process the frame
        const tensor = tf.browser.fromPixels(video)
          .resizeNearestNeighbor([224, 224])
          .toFloat()
          .div(255.0)  // Normalize to [0,1]
          .expandDims();

        // Make a prediction
        const predictions = await model.predict(tensor).data();
        
        // Get the top prediction
        const topPredictionIndex = predictions.indexOf(Math.max(...predictions));
        topPrediction = cardData[topPredictionIndex] || { 
          player_name: 'Test Player', 
          card_set: 'Test Set', 
          card_number: '123' 
        };
        
        setPrediction(topPrediction);
        console.log('Prediction successful:', topPrediction);
        
        // Clean up tensor to prevent memory leaks
        tensor.dispose();
      } catch (error) {
        console.error('Error during detection:', error);
      }

      // Draw the video frame on canvas
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Draw bounding box and label
      if (topPrediction) {
        // Placeholder for bounding box - replace with actual detection if model provides it
        const [x, y, width, height] = [50, 50, video.videoWidth - 100, video.videoHeight - 100]; 
        ctx.strokeStyle = '#00FF00';
        ctx.lineWidth = 4;
        ctx.strokeRect(x, y, width, height);

        ctx.fillStyle = '#00FF00';
        ctx.font = '18px Arial';
        ctx.fillText(
          `${topPrediction.player_name} - ${topPrediction.card_set} #${topPrediction.card_number}`,
          x,
          y > 20 ? y - 10 : y + height + 20
        );
      }
    }
  }, [model, videoRef, canvasRef, cardData]);

  if (!cameraActive) {
    return (
      <Box sx={{ width: '100%', maxWidth: 480, mx: 'auto', mt: 4, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: 320 }}>
        <Paper elevation={3} sx={{ p: 4, width: '100%', textAlign: 'center' }}>
          <Typography variant="h6" gutterBottom>Ready to detect sports cards?</Typography>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Click below to start your camera.
          </Typography>
          <IconButton color="primary" size="large" onClick={() => setCameraActive(true)} sx={{ mt: 2 }}>
            <Box sx={{ 
              width: 64, 
              height: 64, 
              borderRadius: '50%', 
              bgcolor: 'primary.main', 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center',
              color: 'white',
              fontSize: 32
            }}>
              📷
            </Box>
          </IconButton>
          {loading && <CircularProgress sx={{ mt: 2 }} />}
          {error && (
            <Typography color="error" sx={{ mt: 2 }}>
              {error}
            </Typography>
          )}
        </Paper>
      </Box>
    );
  }

  return (
    <Box sx={{ width: '100%', position: 'relative', mt: 2 }}>
      <Box sx={{ position: 'relative', width: '100%', height: 'auto', borderRadius: 2, overflow: 'hidden' }}>
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          style={{ width: '100%', height: 'auto', display: 'block' }}
        />
        <canvas
          ref={canvasRef}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
          }}
        />
      </Box>
      
      <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Button 
          variant="contained" 
          color="secondary" 
          onClick={() => setCameraActive(false)}
        >
          Stop Camera
        </Button>
        
        {prediction && (
          <Paper elevation={2} sx={{ p: 2, flex: 1, ml: 2 }}>
            <Typography variant="subtitle1">
              Detected: {prediction.player_name}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {prediction.card_set} #{prediction.card_number} ({prediction.year})
            </Typography>
          </Paper>
        )}
      </Box>
      
      {error && (
        <Typography color="error" sx={{ mt: 2 }}>
          {error}
        </Typography>
      )}
    </Box>
  );
}

export default App;
