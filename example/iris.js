'use strict';

const iris = require('ml-dataset-iris');
const CV = require('ml-cross-validation');
const SVM = require('..');

// var labels = iris.getClasses();
// const features = iris.getNumbers();
//
// // Separate in 2 categories
// labels = labels.map(label => {
//     if(label === labels[0]) return 1;
//     return -1;
// });

const features = [[20, 10],[30,30]];
var labels = [-1, 1];

const result = CV.leaveOneOut(SVM, features, labels, {
    k: 'radial',
    C: 1,
    par: 1
});
