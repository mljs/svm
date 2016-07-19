# ml-svm

  [![NPM version][npm-image]][npm-url]
  [![build status][travis-image]][travis-url]
  [![David deps][david-image]][david-url]
  [![npm download][download-image]][download-url]
  
Support Vector Machine in Javascript

## Installation

`npm install ml-svm`

## Test

```js
$ npm install
$ npm test
```

## Methods

### new SVM([options])

Creates a new SVM instance with the given parameters or the default ones.

__Arguments__
* `options` - Object with options for the algorithm

__Options__

* `C` - regularization parameter
* `tol` - numerical tolerance
* `max_passes` - max number of times to iterate over alphas without changing
* `k` - the kind of kernel, it could be `linear`, `polynomial` or `radial`
* `par` - parameter used in the polynomial and the radial function of the kernel

__Example__

```js
var SVM = require('ml-svm');

// actually this are the default values
var options = {
  C: 10,
  tol: 10e-2,
  max_passes: 100,
  par: 2,
  k: 'linear'
};

var svm = new SVM(options);
```

### train(X, Y)

Train the SVM with the provided `X` and `Y` training set.

__Arguments__

* `X` - An array of training data point in the form (x1, x2)
* `Y` - An array of training data labels in the domain {1,-1}

__Example__

```js
var X = [[0, 1], [4, 6], [2,0]];
var Y = [-1,1,-1];
var mySvm = new SVM();
mySvm.train(X, Y);
```

### getAlphas()

Returns an array containing the Lagrange multipliers.

### getThreshold()

Returns the threshold of the model function.

### predict([data])

Returns for each data point the predicted label based in the model.

__Arguments__

* `data` - Data point or array of data points.

__Example__

```js
// creates the SVM
var mySvm = new SVM({tol: 0.01});

// train the model
var X = [[0, 1], [4, 6], [2,0]];
var Y = [-1,1,-1];
mySvm.train(X, Y);

// here you have the answer
var ans = mySvm.predict([2,6]);
```

### export()

Exports the model to a JSON object that can be written to disk and reloaded

### load(model)

Returns a new SVM instance based on the `model`.

__Arguments__

* `model` - JSON object generated with `svm.export()`

## Authors

  - [Miguel Asencio](https://github.com/maasencioh)

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
