import React, { Component } from 'react';
import * as tf from '@tensorflow/tfjs';

// Define Model
const model = tf.sequential();
model.add(tf.layers.dense({
    units: 1,
    inputShape:[2],
    activation: 'sigmoid'
}));

// Set optmizer and loss
model.compile({optimizer: tf.train.sgd(0.9), loss: 'meanSquaredError'});

// Generic training data
const xs = tf.tensor2d([[0, 0],[0.5, 0.5],[1, 1]]);
const ys = tf.tensor2d([[1],[0.5],[0]]);

// Train the model using the data
model.fit(xs, ys, {
    epochs: 1000,
    callbacks: {
      onEpochEnd: async (epoch, log) => {
        console.log(`Epoch ${epoch}: loss = ${log.loss}`);
      }
    }
  }
).then(()=>{
    console.log("*** Training Complete ***" );
    console.log("--- Actual Values ---")
    ys.print();
    console.log("--- Prediction ---")
    model.predict(xs).print()
})

export default class App extends Component {
  render() {
    return (
      <div>
        <h1>An area to learn TensorFlow.js</h1>
      </div>
    );
  }
}