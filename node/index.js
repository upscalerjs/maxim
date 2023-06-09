const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const fs = require('fs');

const [modelPath, sampleImage, output] = process.argv.slice(2);

(async () => {
  const [
    image,
    model,
  ] = await Promise.all([
    tf.tidy(() => tf.node.decodePng(fs.readFileSync(path.resolve(__dirname, sampleImage))).expandDims(0).div(255)),
    tf.loadGraphModel(tf.io.fileSystem(path.resolve(__dirname, modelPath))),
  ]);

  if (output) {
    const start = performance.now();
    const pred = model.predict(image);
    console.log(`Node Prediction took ${performance.now() - start}`);
    const contents = getContents(pred);
    fs.writeFileSync(output, JSON.stringify(contents), 'utf-8');
  }
})();

const getContents = (pred) => ({
  shape: pred.shape,
  data: Array.from(pred.dataSync()),
});
