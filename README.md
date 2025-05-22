# North American Bird Classifier for Scrypted

This repository provides a configuration for a North American bird classifier specifically trained on the NABirds dataset, which includes **400 North American bird species**.

## ⚠️ Important Note

Currently, there isn't a publicly hosted TFLite model trained on the NABirds dataset. This repository provides a template configuration that you can use once you have access to such a model.

## About NABirds Dataset

- **400 species** of North American birds
- **48,000 annotated photographs**
- **700 visual categories** (including male, female, and juvenile variations)
- Curated in collaboration with Cornell Lab of Ornithology

## Current Options

### Option 1: Use the Google AIY Birds Classifier Instead
For immediate use, we recommend using our other repository:
- https://github.com/DMontgomery40/bird-classifier-google-aiy
- Identifies 964 bird species globally (includes many North American species)
- Ready to use with Scrypted immediately

### Option 2: Convert Existing PyTorch Model
The [npatta01/Bird-Classifier](https://github.com/npatta01/Bird-Classifier) repository contains a PyTorch model trained on NABirds. To use it with Scrypted:

1. Convert the PyTorch model to TFLite format:
```python
# Example conversion code (requires TensorFlow)
import tensorflow as tf

# Load PyTorch model and convert to ONNX first
# Then convert ONNX to TensorFlow
# Finally convert to TFLite

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

# Save the model
with open('nabirds_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

2. Host the `.tflite` file somewhere accessible (GitHub releases, cloud storage, etc.)
3. Update the `config.json` in this repository with your model URL

### Option 3: Use iNaturalist-based Models
Several bird classifiers are trained on iNaturalist data which includes extensive North American coverage:
- These models use MobileNetV2 architecture
- Available in TFLite format
- Good accuracy for common North American species

## Configuration Structure

The `config.json` file is set up to accept a NABirds TFLite model:

```json
{
  "name": "North American Birds Classifier (NABirds)",
  "version": "1.0.0",
  "model": {
    "url": "YOUR_NABIRDS_TFLITE_MODEL_URL_HERE",
    "type": "tflite"
  },
  "labels": {
    "url": "https://raw.githubusercontent.com/npatta01/Bird-Classifier/master/backend/models/classes.json",
    "type": "json"
  },
  "input": {
    "width": 224,
    "height": 224,
    "channels": 3,
    "mean": [127.5, 127.5, 127.5],
    "std": [127.5, 127.5, 127.5]
  }
}
```

## How to Use (Once You Have a Model)

1. Replace the `model.url` in `config.json` with your TFLite model URL
2. Ensure the labels file matches your model's classes
3. Use this repository URL in Scrypted: `https://github.com/DMontgomery40/bird-classifier-north-america`

## Alternative Bird Detection Resources

- **BirdNET**: Audio-based bird identification (not image-based)
- **eBird**: Cornell's bird observation database
- **Merlin Bird ID**: Mobile app for bird identification

## Contributing

If you successfully convert the NABirds model to TFLite format and host it publicly, please submit a pull request to update this configuration!

## Credits

- NABirds dataset by Cornell Lab of Ornithology
- Original PyTorch implementation by [npatta01](https://github.com/npatta01/Bird-Classifier)