# ml-svm

  [![NPM version][npm-image]][npm-url]
  [![build status][travis-image]][travis-url]
  [![David deps][david-image]][david-url]
  [![npm download][download-image]][download-url]
  
Support Vector Machine in Javascript

## Installation

`npm install ml-svm`

## API
[API documentation](https://mljs.github.io/svm)

## Example

```js
// Instantiate the svm classifier
var SVM = require('ml-svm');

var options = {
  C: 0.01,
  tol: 10e-4,
  maxPasses: 10,
  maxIterations: 10000,
  kernel: 'rbf',
  kernelOptions: {
    sigma: 0.5
  }
};

var svm = new SVM(options);

// Train the classifier - we give him an xor
var features = [[0,0],[0,1],[1,1],[1,0]];
var labels = [1, -1, 1, -1];
svm.train(features, labels);

// Let's see how narrow the margin is
var margins = svm.margin(features);

// Let's see if it is separable by testing on the training data
svm.predict(features); // [1, -1, 1, -1]

// I want to see what my support vectors are
var supportVectors = svm.supportVectors();
 
// Now we want to save the model for later use
var model = svm.toJSON();

/// ... later, you can make predictions without retraining the model
var importedSvm = SVM.load(model);
importedSvm.predict(features); // [1, -1, 1, -1] 
```


## Authors

  - [Miguel Asencio](https://github.com/maasencioh)
  - [Daniel Kostro](https://github.com/stropitek)

## License

  [MIT](./LICENSE)

[npm-image]: https://img.shields.io/npm/v/ml-svm.svg?style=flat-square
[npm-url]: https://npmjs.org/package/ml-svm
[travis-image]: https://img.shields.io/travis/mljs/svm/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/mljs/svm
[david-image]: https://img.shields.io/david/mljs/svm.svg?style=flat-square
[david-url]: https://david-dm.org/mljs/svm
[download-image]: https://img.shields.io/npm/dm/ml-svm.svg?style=flat-square
[download-url]: https://npmjs.org/package/ml-svm
