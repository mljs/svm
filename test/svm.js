var svm = require('../src/svm.js');

var X = [[0, 1], [4, 6], [2,0]];
var Y = [-1,1,-1];

describe('SVM test', function () {

    it('test b threshold', function () {
        var ans = svm(X, Y);
        console.log(ans);
        ans.b.should.be.approximately(-1.2,0.2);
    });
});