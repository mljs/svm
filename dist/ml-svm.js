/**
 * ml-svm - Support Vector Machine in Javascript
 * @version v1.0.1
 * @link https://github.com/mljs/svm
 * @license MIT
 */
!function(e){if("object"==typeof exports&&"undefined"!=typeof module)module.exports=e();else if("function"==typeof define&&define.amd)define([],e);else{var f;"undefined"!=typeof window?f=window:"undefined"!=typeof global?f=global:"undefined"!=typeof self&&(f=self),f.mlSvm=e()}}(function(){var define,module,exports;return (function e(t,n,r){function s(o,u){if(!n[o]){if(!t[o]){var a=typeof require=="function"&&require;if(!u&&a)return a(o,!0);if(i)return i(o,!0);var f=new Error("Cannot find module '"+o+"'");throw f.code="MODULE_NOT_FOUND",f}var l=n[o]={exports:{}};t[o][0].call(l.exports,function(e){var n=t[o][1][e];return s(n?n:e)},l,l.exports,e,t,n,r)}return n[o].exports}var i=typeof require=="function"&&require;for(var o=0;o<r.length;o++)s(r[o]);return s})({1:[function(require,module,exports){
exports.svm = require('./svm');
exports.kernel = require('./kernel');
},{"./kernel":2,"./svm":3}],2:[function(require,module,exports){
'use strict';

/**
 * Kernel function to return the dot product for different spaces
 * @param {Array <number>} x1 - input first vector
 * @param {Array <number>} x2 - input second vector
 * @param {string} func - the kind of transformation
 * @param {number} par - parameter used in the polynomial and the radial function
 * @return {number} calculus of the dot product using the function
 * */
function kernel(x1,x2,func,par) {
    func = (typeof func === 'undefined') ? 'lineal' : func;
    par = (typeof par === 'undefined') ? 2 : par;

    var p = dot(x1,x2);
    if (func === 'lineal'){
        return p;
    }
    else if(func === 'polynomial') {
        return Math.pow((p + 1), par);
    }
    else if(func === 'radial') {
        var l = x1.length;
        var rest = new Array(l);
        for (var i = 0; i < l; i++) {
            rest[i] = x1[i] - x2[i];
        }
        var norm = dot(rest, rest);
        return Math.exp((norm)/(-2*par*par));
    }
    else {
        throw new TypeError('Function kernel undefined');
    }
}

/**
 * The dot product between the p1 and p2 vectors
 * @param {Array <number>} p1 - first vector to get dot product
 * @param {Array <number>} p2 - second vector to get dot product
 * @returns {number} dot product between the p1 and p2 vectors
 */
function dot(p1, p2) {
    if (p1.length !== p2.length) {
        throw new TypeError('Arrays should have the same length');
    }
    var l = p1.length;
    var prod = 0;

    for (var i = 0; i < l; i++) {
        prod += p1[i] * p2[i];
    }

    return prod;
}

module.exports = kernel;
},{}],3:[function(require,module,exports){
'use strict';
var kernel = require("./kernel");

/**
 * Parameters to implement function
 * @type {{C: number, tol: number, max_passes: number, par: number, k: string}}
 * @param {number} C - regularization parameter
 * @param {number} tol - numerical tolerance
 * @param {number} max_passes - max number of times to iterate over alphas without
 * changing
 * @param {string} k - the kind of kernel
 * @param {number} par - parameter used in the polynomial and the radial function
 * of the kernel
 */
var defaultOptions = {
    C: 10,
    tol: 10e-2,
    max_passes: 100,
    par: 2,
    k: 'lineal'
};

/**
 * Function to calculate the estimated prediction
 * @param {Array <number>} x - point where calculate the function prediction
 * @param {Array <Array <number>>} X - training data point in the form (x1, x2)
 * @param {Array <number>} Y - training data labels in the domain {1,-1}
 * @param {Array <number>} alpha - Lagrange multipliers
 * @param {number} b - threshold of the function
 * @param {string} k - the kind of kernel
 * @param {number} par - parameter used in the polynomial and the radial function
 * of the kernel
 * @returns {number}
 */
function f(x, X, Y, alpha, b, k, par) {
    var m = X.length;
    var aux = b;
    for (var i = 0; i < m; i++) {
        b += alpha[i]*Y[i]*kernel(X[i],x, k, par)
    }
    return aux;
}

/**
 * Simplified version of the Sequential Minimal Optimization algorithm for training
 * support vector machines
 * @param {{json}} options - parameters to implement function
 * @constructor
 */
function SVM(options) {
    options = options || {};
    this.options = {};
    for (var o in defaultOptions) {
        if (options.hasOwnProperty(o)) {
            this.options[o] = options[o];
        } else {
            this.options[o] = defaultOptions[o];
        }
    }
    this.b = 0;
}

/**
 * Train the SVM model
 * @param {Array <Array <number>>} X - training data point in the form (x1, x2)
 * @param {Array <number>} Y - training data labels in the domain {1,-1}
 */
SVM.prototype.train = function (X, Y) {
    var m = Y.length;
    var alpha = new Array(m);
    for (var a = 0; a < m; a++)
        alpha[a] = 0;
    if (X.length !== m)
        throw new TypeError('Arrays should have the same length');
    var b = 0,
        b1 = 0,
        b2 = 0,
        iter = 0,
        Ei = 0,
        Ej = 0,
        ai = 0,
        aj = 0,
        L = 0,
        H = 0,
        eta = 0;

    while (iter < this.options.max_passes) {
        var numChange = 0;
        for (var i = 0; i < m; i++) {
            Ei = f(X[i],X,Y,alpha,b,this.options.k,this.options.par) - Y[i];
            if (((Y[i]*Ei < -this.options.tol) && (alpha[i] < this.options.C)) || ((Y[i]*Ei > this.options.tol) && (alpha[i] > 0))) {
                var j = 0;
                do {
                    j = Math.ceil(Math.random()*(m - 1));
                }
                while (j === i);
                Ej = f(X[j],X,Y,alpha,b,this.options.k,this.options.par) - Y[j];
                ai = alpha[i];
                aj = alpha[j];
                if (Y[i] === Y[j]) {
                    L = Math.max(0, ai+aj-this.options.C);
                    H = Math.min(this.options.C, ai+aj);
                }
                else  {
                    L = Math.max(0, ai-aj);
                    H = Math.min(this.options.C, this.options.C-ai+aj);
                }
                if (L !== H) {
                    eta = 2*kernel(X[i],X[j], this.options.k, this.options.par) - kernel(X[i],X[i], this.options.k, this.options.par) - kernel(X[j],X[j], this.options.k, this.options.par);
                    if (eta < 0) {
                        alpha[j] = alpha[j] - (Y[j]*(Ei - Ej)) / eta;
                        if (alpha[j] > H)
                            alpha[j] = H;
                        else if (alpha[j] < L)
                            alpha[j] = L;
                        if (Math.abs(aj - alpha[j]) >= 10e-5) {
                            alpha[i] = alpha[i] + Y[i]*Y[j]*(aj - alpha[j]);
                            b1 = b - Ei - Y[i]*(alpha[i] - ai)*kernel(X[i],X[i], this.options.k, this.options.par) - Y[j]*(alpha[j] - aj)*kernel(X[i],X[j], this.options.k, this.options.par);
                            b2 = b - Ej - Y[i]*(alpha[i] - ai)*kernel(X[i],X[j], this.options.k, this.options.par) - Y[j]*(alpha[j] - aj)*kernel(X[j],X[j], this.options.k, this.options.par);
                            if ((alpha[i] < this.options.C) && (alpha[i] > 0))
                                b = b1;
                            else if ((alpha[j] < this.options.C) && (alpha[j] > 0))
                                b = b2;
                            else
                                b = (b1 + b2) / 2;
                            numChange += 1;
                        }
                    }
                }
            }
        }
        if (numChange == 0)
            iter += 1;
        else
            iter = 0;
    }
    this.b = b;
    var s = X[0].length;
    this.W = new Array(s);
    for (var r = 0; r < s; r++) {
        this.W[r] = 0;
        for (var w = 0; w < m; w++)
            this.W[r] += Y[w]*alpha[w]*X[w][r];
    }
    this.alphas = alpha.splice();
};

/**
 * Recreates a SVM based in the exported model
 * @param {{name: string, ,options: {json} ,alpha: Array<number>, b: number}} model
 * @returns {SVM}
 */
SVM.load = function (model) {
    if (model.name === 'SVM') {
        var svm = new SVM(model.options);
        svm.W = model.W.slice();
        svm.b = model.b;
        return svm;
    } else {
        throw new TypeError('expecting a SVM model');
    }
};

/**
 * Let's have a JSON to recreate the model
 * @returns {{name: String("SVM"), ,options: {json} ,alpha: Array<number>, b: number}}
 * name identifier, options to recreate model, the Lagrange multipliers and the
 * threshold of the objective function
 */
SVM.prototype.export = function () {
    var model = {
        name: 'SVM'
    };
    model.options = this.options;
    model.W = this.W;
    model.b = this.b;
    return model;
};

/**
 * Return the Lagrange multipliers
 * @returns {Array <number>}
 */
SVM.prototype.getAlphas = function () {
    return this.alphas.slice();
};

/**
 * Returns the threshold of the model function
 * @returns {number} threshold of the function
 */
SVM.prototype.getThreshold = function () {
    return this.b;
};

/**
 * Use the train model to make predictions
 * @param {Array} p - An array or a single dot to have the prediction
 * @returns {*} An array or a single {-1, 1} value of the prediction
 */
SVM.prototype.predict = function (p) {
    var ev;
    if (Array.isArray(p) && (Array.isArray(p[0]) || (typeof p[0] === 'object'))) {
        var ans = new Array(p.length);
        for (var i = 0; i < ans.length; i++) {
            ev = this.b;
            for (var j = 0; j < this.W.length; j++)
                ev += this.W[j]*p[j];
            if (ev < 0)
                ans[i] = -1;
            else
                ans[i] = 1;
        }
        return ans;
    }
    else {
        ev = this.b;
        for (var e = 0; e < this.W.length; e++)
            ev += this.W[e]*p[e];
        if (ev < 0)
            return -1;
        else
            return 1;
    }
};

module.exports = SVM;
},{"./kernel":2}]},{},[1])(1)
});