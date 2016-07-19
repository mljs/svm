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
    return getKernel(func)(x1, x2, par);
}

/**
 * The dot product between the p1 and p2 vectors
 * @param {Array <number>} p1 - first vector to get dot product
 * @param {Array <number>} p2 - second vector to get dot product
 * @returns {number} dot product between the p1 and p2 vectors
 */
function dot(p1, p2) {
    var l = p1.length;
    var prod = 0;

    for (var i = 0; i < l; i++) {
        prod += p1[i] * p2[i];
    }

    return prod;
}

function getKernel(func) {
    func = (typeof func === 'undefined') ? 'linear' : func;

    switch(func) {
        case 'linear':
            return kernellinear;
        case 'polynomial':
            return kernelPolynomial;
        case 'radial':
            return kernelRadial;
        default:
            throw new TypeError('Function kernel undefined: ' + func);
    }
}

function kernellinear(x1,x2) {
    return dot(x1,x2);
}

function kernelPolynomial(x1, x2, par) {
    par = (typeof par === 'undefined') ? 2 : par;
    return Math.pow((dot(x1, x2) + 1), par);
}

function kernelRadial(x1, x2, par) {
    par = (typeof par === 'undefined') ? 2 : par;
    var l = x1.length;
    var rest = new Array(l);
    for (var i = 0; i < l; i++) {
        rest[i] = x1[i] - x2[i];
    }
    var norm = dot(rest, rest);
    return Math.exp((norm)/(-2*par*par));
}

module.exports = {
    kernel: kernel,
    getKernel: getKernel,
    linear : kernellinear,
    polynomial : kernelPolynomial,
    radial : kernelRadial
};
