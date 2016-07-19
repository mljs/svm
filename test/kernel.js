'use strict';

var kernel = require('..').kernel;

var v1 = [0, 1, 4, 6, 2];
var v2 = [0, 2, 5, 6, 7];

describe('kernel test', function () {

    it('no change product', function () {
        kernel(v1,v2).should.equal(72);
        kernel(v1,v2,"linear").should.equal(72);
    });

    it('polynomial product', function () {
        kernel(v1,v2,"polynomial").should.equal(5329);
        kernel(v1,v2,"polynomial",2).should.equal(5329);
        kernel(v1,v2,"polynomial",3).should.equal(389017);
    });

    it('radial product', function () {
        kernel(v1,v2,"radial").should.be.approximately(0.03421811831,0.0001);
        kernel(v1,v2,"radial",2).should.be.approximately(0.03421811831,0.0001);
        kernel(v1,v2,"radial",3).should.be.approximately(0.22313016014,0.0001);
    });


});