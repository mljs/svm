'use strict';
var SVM = require('..');
var xsadd = require('ml-xsadd');
var random = new xsadd(0).random;

var data = {
    linear1: {
        features: [[0, -200], [400, 600], [200, 0]],
        labels: [-1, 1, -1]
    },
    linear2: {
        features: [[0, -200], [1, 600], [1 / 2, 0]],
        labels: [-1, 1, -1]
    },
    linear3: {
        features: [[0, 1], [4, 6], [2, 0]],
        labels: [-1, 1, -1]
    },
    xor: {
        features: [[0, 0], [0, 1], [1, 1], [1, 0]],
        labels: [1, -1, 1, -1]
    }
};
describe('SVM', function () {
    it('should solve a linearly separable case', function () {
        var features = data.linear1.features;
        var labels = data.linear1.labels;
        var svm = new SVM({random});
        svm.train(features, labels);
        svm.predict(features).should.eql(labels);
        svm.predict(features[0]).should.eql(labels[0]);
        svm.supportVectors().should.eql([1, 2]);
    });

    it('should reload the linear model', function () {
        var features = data.linear3.features;
        var labels = data.linear3.labels;
        var svm = new SVM({random});
        svm.train(features, labels);
        var exp = JSON.parse(JSON.stringify(svm));
        var reloadedSvm = SVM.load(exp);
        reloadedSvm.predict(features).should.eql(labels);
        (function () {
            reloadedSvm.supportVectors();
        }).should.throw(/Cannot get support vectors from saved linear model/);
    });

    it('should solve a linearly separable case without whitening', function () {
        var features = data.linear3.features;
        var labels = data.linear3.labels;
        var svm = new SVM({
            random,
            whitening: false
        });
        svm.train(features, labels);
        svm.predict(features).should.eql(labels);
    });

    it('Some cases are not separable without whitening', function () {
        var features = data.linear2.features;
        var labels = data.linear2.labels;
        var svm = new SVM({
            random
        });
        var svm1 = new SVM({
            random,
            whitening: false
        });

        svm.train(features, labels);
        svm1.train(features, labels);

        svm.predict(features).should.eql(labels);
        svm1.predict(features).should.not.eql(labels);
    });


    it('should solve xor with rbf', function () {
        var svm = new SVM({
            kernel: 'rbf',
            kernelOptions: {
                sigma: 0.1
            },
            random
        });
        //var features = data.xor.features;
        var width = 200, height = 200;
        var features = [[width / 4, height / 4], [3 * width / 4, height / 4], [3 * width / 4, 3 * height / 4], [width / 4, 3 * height / 4]];
        // var labels = data.xor.labels;
        var labels = [-1, 1, -1, 1];
        svm.train(features, labels);
        svm.predict(features).should.eql(labels);
    });

    it('should solve xor with reloaded model', function () {
        var svm = new SVM({
            kernel: 'rbf',
            kernelOptions: {
                sigma: 0.1
            },
            random
        });
        var features = data.xor.features;
        var labels = data.xor.labels;
        svm.train(features, labels);
        var model = JSON.parse(JSON.stringify(svm));
        var reloadedSvm = SVM.load(model);
        reloadedSvm.predict(features).should.eql(labels);
    });
});
