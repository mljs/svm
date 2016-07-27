'use strict';

var SVM = require('..');

var options = {
    kernel: 'linear',
    C:1,
};

var svm =  new SVM(options);

const features = [[0,2],[0,0],[2,2],[2,0]];
var labels = [-1, -1, 1, 1];

// var features = [[0,0],[0,1],[1,1],[1,0]];
// var labels = [1, -1, 1, -1];

for(var i=0; i<1000; i++) {
    svm.train(features, labels);
    svm.predict(features);
}

