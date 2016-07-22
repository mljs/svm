'use strict';
var SVM = require('..');
var data = {
    linear: {
        features: [[0, -2], [4, 6], [2,0]],
        labels: [-1,1,-1]
    },
    xor: {
        features: [[0,0],[0,1],[1,1],[1,0]],
        labels: [1, -1, 1, -1]
    }
};
describe('SVM', function () {
    it('should solve a linearly separable case', function () {
        var features = data.linear.features;
        var labels = data.linear.labels;
        var svm = new SVM();
        svm.train(features, labels);
        svm.predict(features).should.eql(labels);
        svm.predict(features[0]).should.eql(labels[0]);
        // Linearly separable case = 1 support vector for each of the two classes
        svm.supportVectors().should.eql([features[1], features[2]]);
    });

    it('should reload the linear model', function () {
        var features = [[0, 1], [4, 6], [2,0]];
        var labels = [-1,1,-1];
        var svm = new SVM();
        svm.train(features, labels);
        var exp = JSON.parse(JSON.stringify(svm));
        var reloadedSvm = SVM.load(exp);
        reloadedSvm.predict(features).should.eql(labels);
        (function() {
            reloadedSvm.supportVectors();
        }).should.throw(/Cannot get support vectors from saved linear model/)
    });

    it('should solve xor with rbf', function () {
        var svm = new SVM({
            kernel: 'rbf',
            kernelOptions: {
                sigma: 0.5
            }
        });
        var features = data.xor.features;
        var labels = data.xor.labels;
        svm.train(features, labels);
        svm.predict(features).should.eql(labels);
    });

    it('should solve xor with reloaded model', function () {
        var svm = new SVM({
            kernel: 'rbf',
            kernelOptions: {
                sigma: 0.5
            }
        });
        var features = data.xor.features;
        var labels = data.xor.labels;
        svm.train(features, labels);
        var model = JSON.parse(JSON.stringify(svm));
        var reloadedSvm = SVM.load(model);
        reloadedSvm.predict(features).should.eql(labels);
    })
});
