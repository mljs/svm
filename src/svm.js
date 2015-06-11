'use strict';

var kernel = require("./kernel");
/**
 * Simplified version of the Sequential Minimal Optimization algorithm for training
 * support vector machines
 * @param {Array <Array <number>>} X - training data point in the form (x1, x2)
 * @param {Array <number>} Y - training data labels in the domain {1,-1}
 * @param {number} C - regularization parameter
 * @param {number} tol - numerical tolerance
 * @param {number} max_passes - max number of times to iterate over alphas without
 * changing
 * @param {string} k - the kind of kernel
 * @param {number} par - parameter used in the polynomial and the radial function
 * of the kernel
 * @returns {{alphas: Array<number>, b: number}} returns the Lagrange multipliers and
 * the threshold of the objective function
 */
module.exports = function svm(X, Y, C, tol, max_passes, k, par) {
    var m = Y.length;
    if (X.length !== m) return 0;
    max_passes = (typeof max_passes === 'undefined') ? 100 : max_passes;
    tol = (typeof tol === 'undefined') ? 10e-2 : tol;
    k = (typeof k === 'undefined') ? 'lineal' : k;
    par = (typeof par === 'undefined') ? 2 : par;
    C = (typeof C === 'undefined') ? 10 : C;
    var alpha = new Array(m);
    for (var i = 0; i < m; i++)
        alpha[i] = 0;
    var b = 0;
    var b1 = 0;
    var b2 = 0;
    var iter = 0;
    var Ei = 0;
    var Ej = 0;
    var ai = 0;
    var aj = 0;
    var L = 0;
    var H = 0;
    var eta = 0;

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
    var f = function (x, X, Y, alpha, b, k, par) {
        var m = X.length;
        var aux = b;
        for (var i = 0; i < m; i++) {
            b += alpha[i]*Y[i]*kernel(X[i],x, k, par)
        }
        return aux;
    };

    while (iter < max_passes) {
        var numChange = 0;
        for (var i = 0; i < m; i++) {
            Ei = f(X[i],X,Y,alpha,b,k,par) - Y[i];
            if (((Y[i]*Ei < -tol) && (alpha[i] < C)) || ((Y[i]*Ei > tol) && (alpha[i] > 0))) {
                var j = 0;
                do {
                    j = Math.ceil(Math.random()*(m - 1));
                }
                while (j === i);
                Ej = f(X[j],X,Y,alpha,b,k,par) - Y[j];
                ai = alpha[i];
                aj = alpha[j];
                if (Y[i] === Y[j]) {
                    L = Math.max(0, ai+aj-C);
                    H = Math.min(C, ai+aj);
                }
                else  {
                    L = Math.max(0, ai-aj);
                    H = Math.min(C, C-ai+aj);
                }
                if (L !== H) {
                    eta = 2*kernel(X[i],X[j], k, par) - kernel(X[i],X[i], k, par) - kernel(X[j],X[j], k, par);
                    if (eta < 0) {
                        alpha[j] = alpha[j] - (Y[j]*(Ei - Ej)) / eta;
                        if (alpha[j] > H)
                            alpha[j] = H;
                        else if (alpha[j] < L)
                            alpha[j] = L;
                        if (Math.abs(aj - alpha[j]) >= 10e-5) {
                            alpha[i] = alpha[i] + Y[i]*Y[j]*(aj - alpha[j]);
                            b1 = b - Ei - Y[i]*(alpha[i] - ai)*kernel(X[i],X[i], k, par) - Y[j]*(alpha[j] - aj)*kernel(X[i],X[j], k, par);
                            b2 = b - Ej - Y[i]*(alpha[i] - ai)*kernel(X[i],X[j], k, par) - Y[j]*(alpha[j] - aj)*kernel(X[j],X[j], k, par);
                            if ((alpha[i] < C) && (alpha[i] > 0))
                                b = b1;
                            else if ((alpha[j] < C) && (alpha[j] > 0))
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
    return {
        "alphas": alpha,
        "b": b
    };
};