const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const fs = require('fs');

const [modelPath, sampleImage, output] = process.argv.slice(2);

const { Einsum, backend_util, util } = require('@tensorflow/tfjs-core');
const reshape = require('@tensorflow/tfjs-node/dist/kernels/Reshape').reshapeConfig.kernelFunc;
const transpose = require('@tensorflow/tfjs-node/dist/kernels/Transpose').transposeConfig.kernelFunc;
const multiply = require('@tensorflow/tfjs-node/dist/kernels/Multiply').multiplyConfig.kernelFunc;
const sum = require('@tensorflow/tfjs-node/dist/kernels/Sum').sumConfig.kernelFunc;
const { registerKernel } = require('@tensorflow/tfjs-core');

function einsum(args) {
  // console.log('args 1', args);
  const { inputs, backend, attrs } = args;
  const { equation } = attrs;
  const tensors = inputs;

  const { allDims, summedDims, idDims } = backend_util.decodeEinsumEquation(equation, tensors.length);
  backend_util.checkEinsumDimSizes(allDims.length, idDims, tensors);
  const { path, steps } = backend_util.getEinsumComputePath(summedDims, idDims);

  const nSteps = steps.length;
  let out = null;
  let numDimsRemaining = allDims.length;
  const tensorsToDispose = [];
  for (let i = 0; i < nSteps; ++i) {
    for (const idTerm of steps[i]) {
      const { permutationIndices: perm, expandDims: dimsToExpand } =
        backend_util.getEinsumPermutation(numDimsRemaining, idDims[idTerm]);
      let x;
      if (backend_util.isIdentityPermutation(perm)) {
        x = tensors[idTerm];
      } else {
        x = transpose({ inputs: { x: tensors[idTerm] }, backend, attrs: { perm } });
        tensorsToDispose.push(x);
      }
      const targetShape = x.shape.slice();
      for (let k = 0; k < dimsToExpand.length; ++k) {
        targetShape.splice(dimsToExpand[k], 0, 1);
      }

      if (!util.arraysEqual(x.shape, targetShape)) {
        // console.log('reshape!');
        x = reshape({ inputs: { x }, backend, attrs: { shape: targetShape } });
        tensorsToDispose.push(x);
      }
      if (out === null) {
        out = x;
      } else {
        // tslint:disable-next-line: no-unnecessary-type-assertion
        out = multiply({ inputs: { a: x, b: out }, backend });
        tensorsToDispose.push(out);
      }
    }
    if (i < nSteps - 1) {
      if (path[i] >= 0) {
        out = sum({
          inputs: { x: out },
          backend,
          attrs: {
            axis: path[i] - (allDims.length - numDimsRemaining),
            keepDims: false
          }
        });
        tensorsToDispose.push(out);
      }
      numDimsRemaining--;
    }
  }

  // Clean up intermediate tensors.
  for (const tensorInfo of tensorsToDispose) {
    if (tensorInfo === out) {
      continue;
    }
    // console.log(backend);
    backend.disposeData(tensorInfo);
  }

  return out;
}


registerKernel({
  // ...einsumConfig,
  kernelName: Einsum,
  backendName: 'tensorflow',
  kernelFunc: einsum,
});

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
    const start = performance.now();
    const pred = model.predict(image);
    console.log(`Prediction took ${performance.now() - start}`);
    const contents = getContents(pred);
    fs.writeFileSync(output, JSON.stringify(contents), 'utf-8');
  }
})();

const getContents = (pred) => ({
  shape: pred.shape,
  data: Array.from(pred.dataSync()),
});
