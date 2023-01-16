const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const fs = require('fs');

const [modelPath, sampleImage, output] = process.argv.slice(2);

const loadModel = async (filename) => {
  const modelPath = tf.io.fileSystem(path.resolve(__dirname, filename));
  return tf.loadGraphModel(modelPath);
};
const ScaleAndTranslate = ({ inputs }) => {
  const [input, size_tensor] = inputs;
  return tf.tidy(() => {
    return tf.image.resizeNearestNeighbor(
      input,
      size_tensor.dataSync(),
    )
  })
}

tf.registerOp('ScaleAndTranslate', ScaleAndTranslate);

const loadImage = (imagePath) => tf.tidy(() => tf.node.decodePng(fs.readFileSync(imagePath)).expandDims(0).div(255));

(async () => {
  const imagePath = path.resolve(__dirname, sampleImage);
  const [
    image,
    model,
  ] = await Promise.all([
    loadImage(imagePath),
    loadModel(modelPath),
  ]);
  if (output) {
    const pred = model.predict(image);
    const contents = getContents(pred);
    fs.writeFileSync(output, JSON.stringify(contents), 'utf-8');
  }
})();

const getContents = (pred) => ({
  shape: pred.shape,
  data: Array.from(pred.dataSync()),
});
