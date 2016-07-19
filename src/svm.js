'use strict';
const Kernel = require('ml-kernel');

/**
 * Parameters to implement function
 * @type {{C: number, tol: number, max_passes: number, par: number, kernel: string}}
 * @param {number} C - regularization parameter
 * @param {number} tol - numerical tolerance
 * @param {number} max_passes - max number of times to iterate over alphas without
 * changing
 * @param {string} kernel - the kind of kernel
 * @param {number} par - parameter used in the polynomial and the radial function
 * of the kernel
 */
var defaultOptions = {
    C: 10,
    tol: 10e-2,
    max_passes: 10,
    par: 2,
    kernel: 'linear'
};

/**
 * Simplified version of the Sequential Minimal Optimization algorithm for training
 * support vector machines
 * @param {{Object}} options - parameters to implement function
 * @constructor
 */
function SVM(options) {
    this.options = Object.assign({}, defaultOptions, options);

    this.kernel = new Kernel(this.options.kernel, options.kernelOptions);
    this.b = 0;
}

/**
 * Train the SVM model
 * @param {Array <Array <number>>} X - training data point in the form (x1, x2)
 * @param {Array <number>} Y - training data labels in the domain {1,-1}
 */
SVM.prototype.train = function (X, Y) {
    this.N = Y.length;
    this.D = X[0].length;
    this.X = X;
    this.Y = Y;
    this.b = 0;
    this.W = undefined;

    var kernel = this.kernel.compute(X);
    var m = Y.length;
    var alpha = new Array(m).fill(0);
    this.alphas = alpha;
    for (var a = 0; a < m; a++)
        alpha[a] = 0;
    if (X.length !== m)
        throw new TypeError('Arrays should have the same length');
    var b1 = 0,
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
            Ei = this.marginOne(X[i]) - Y[i];
            if (((Y[i]*Ei < -this.options.tol) && (alpha[i] < this.options.C)) || ((Y[i]*Ei > this.options.tol) && (alpha[i] > 0))) {
                var j = i;
                while(j===i) j=randi(0, m);
                Ej = this.marginOne(X[j]) - Y[j];
                ai = alpha[i];
                aj = alpha[j];
                if (Y[i] === Y[j]) {
                    L = Math.max(0, ai+aj-this.options.C);
                    H = Math.min(this.options.C, ai+aj);
                }
                else  {
                    L = Math.max(0, aj-ai);
                    H = Math.min(this.options.C, this.options.C+aj+ai);
                }
                if(Math.abs(L - H) < 1e-4) continue;

                eta = 2*kernel[i][j] - kernel[i][i] - kernel[j][j];
                if(eta >=0) continue;
                var newaj = alpha[j] - (Y[j]*(Ei - Ej)) / eta;
                alpha[j] = alpha[j] - (Y[j]*(Ei - Ej)) / eta;
                if (newaj > H)
                    newaj = H;
                else if (newaj < L)
                    newaj = L;
                if(Math.abs(aj - newaj) < 10e-4) continue;
                alpha[j] = newaj;
                alpha[i] = alpha[i] + Y[i]*Y[j]*(aj - newaj);
                b1 = this.b - Ei - Y[i]*(alpha[i] - ai)*kernel[i][i] - Y[j]*(alpha[j] - aj)*kernel[i][i];
                b2 = this.b - Ej - Y[i]*(alpha[i] - ai)*kernel[i][j] - Y[j]*(alpha[j] - aj)*kernel[j][j];
                this.b = (b1 + b2) / 2;
                if (alpha[i] < this.options.C && alpha[i] > 0) this.b = b1;
                if (alpha[j] < this.options.C && alpha[j] > 0) this.b = b2;
                numChange += 1;
            }
        }
        if (numChange == 0)
            iter += 1;
        else
            iter = 0;
    }
    var s = X[0].length;
    this.W = new Array(s);
    for (var r = 0; r < s; r++) {
        this.W[r] = 0;
        for (var w = 0; w < m; w++)
            this.W[r] += Y[w]*alpha[w]*X[w][r];
    }
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
    if(Array.isArray(p) && Array.isArray(p[0])) {
        return p.map(this.predictOne.bind(this));
    } else {
        return this.predictOne(p);
    }
};

SVM.prototype.margin = function(p) {
    if(Array.isArray(p)) {
        return p.map(this.marginOne.bind(this));
    } else {
        return this.marginOne(p);
    }
};


SVM.prototype.marginOne = function(p) {
    var ans = this.b;
    if(this.options.kernel === 'linear' && this.W) {
        // Use weights, it's faster
        for(var i=0; i<this.W.length; i++) {
            ans += this.W[i] * p[i];
        }
    } else {
        for(var i=0; i<this.N; i++) {
            ans += this.alphas[i] * this.Y[i] * this.kernel.compute([p], [this.X[i]])[0][0];
        }
    }
    return ans;
};



SVM.prototype.predictOne = function(p) {
    var margin = this.marginOne(p);
    return margin > 0 ? 1 : -1;
};

function randi(a, b) {
    return Math.floor(Math.random()*(b-a)+a);
}

module.exports = SVM;