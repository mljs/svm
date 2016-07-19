'use strict';
var SVM = require('..');

var X = [[0, 1], [4, 6], [2,0]];
var Y = [-1,1,-1];

describe('SVM', function () {

    it('test train, predict, export and load', function () {
        var mySvm = new SVM({tol: 0.01});
        mySvm.train(X, Y);
        mySvm.predict([2,6]).should.equal(1);
        mySvm.predict([[2,6]])[0].should.equal(1);
        var exp = mySvm.export();
        var reloadedSvm = SVM.load(exp);
        reloadedSvm.predict([2,6]).should.equal(1);
    });

    it('solve xor with rbf', function () {
        var svm = new SVM({
            kernel: 'rbf',
            kernelOptions: {
                sigma: 0.5
            }
        });
        var features = [[0,0],[0,1],[1,1],[1,0]];
        var labels = [1, -1, 1, -1];
        svm.train(features, labels);
        svm.predict([[0,0],[0,1],[1,1],[1,0]]).should.deepEqual(labels);
    });
});