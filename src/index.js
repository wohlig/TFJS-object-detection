import React, { useState } from "react";
import ReactDOM from "react-dom";
import * as tf from "@tensorflow/tfjs";
import _ from "lodash";
import moment from "moment";
import "./styles.css";
tf.setBackend("webgl");

const threshold = 0.4;
var startTime = moment();

// async function load_model() {
//   const model = await loadGraphModel(
//     "https://raw.githubusercontent.com/souvikwohlig/image_classifier_diamond/main/assets/model.json"
//   );
//   return model;
// }

async function load_model() {
  const model = await tf.loadGraphModel(
    "https://raw.githubusercontent.com/wohlig/TFJS-object-detection/master/models/diamond-detector/model.json"
  );
  return model;
}

class App extends React.Component {
  videoRef = React.createRef();
  canvasRef = React.createRef();
  constructor(props) {
    super(props);
    this.state = {
      count: 0,
    };
  }
  componentDidMount() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      const webCamPromise = navigator.mediaDevices
        .getUserMedia({
          audio: false,
          video: {
            facingMode: "user",
          },
        })
        .then((stream) => {
          window.stream = stream;
          this.videoRef.current.srcObject = stream;
          return new Promise((resolve, reject) => {
            this.videoRef.current.onloadedmetadata = () => {
              resolve();
            };
          });
        });

      const modelPromise = load_model();

      Promise.all([modelPromise, webCamPromise])
        .then((values) => {
          this.detectFrame(this.videoRef.current, values[0]);
        })
        .catch((error) => {
          console.error(error);
        });
    }
  }

  detectFrame = (video, model) => {
    tf.engine().startScope();
    startTime = moment();
    model.executeAsync(this.process_input(video)).then((prediction) => {
      var boxes = prediction[2].dataSync();
      boxes = _.map(boxes, function(value) {
        return value;
      });
      boxes = _.chunk(boxes, 4);

      var scores = prediction[6].dataSync();

      var finalResponse = _.map(scores, function(score, key) {
        var retObj = {
          score: score,
          box: boxes[key],
        };
        return retObj;
      });
      finalResponse = _.filter(finalResponse, function(obj) {
        return obj.score >= threshold;
      });
      console.log(finalResponse);
      this.setState({ count: finalResponse.length });
      this.renderPredictions(finalResponse, video);
      requestAnimationFrame(() => {
        startTime = moment();
        var reactApp = this;
        setTimeout(function() {
          reactApp.detectFrame(video, model);
        }, 2000);
      });
      tf.engine().endScope();
    });
  };

  process_input(video_frame) {
    const tfimg = tf.browser.fromPixels(video_frame).toInt();
    const expandedimg = tfimg.transpose([0, 1, 2]).expandDims();
    return expandedimg;
  }

  buildDetectedObjects(response) {
    var video_frame = document.getElementById("frame");

    const detectionObjects = _.map(response, function(obj) {
      const bbox = [];
      const minY = obj.box[0] * video_frame.offsetHeight;
      const minX = obj.box[1] * video_frame.offsetWidth;
      const maxY = obj.box[2] * video_frame.offsetHeight;
      const maxX = obj.box[3] * video_frame.offsetWidth;
      bbox[0] = minX;
      bbox[1] = minY;
      bbox[2] = maxX - minX;
      bbox[3] = maxY - minY;
      return {
        score: obj.score,
        bbox: bbox,
      };
    });
    console.log(detectionObjects);
    return detectionObjects;
  }

  renderPredictions = (response) => {
    const ctx = this.canvasRef.current.getContext("2d");
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    // Font options.
    const font = "16px sans-serif";
    ctx.font = font;
    ctx.textBaseline = "top";

    const detections = this.buildDetectedObjects(response);

    detections.forEach((item) => {
      const x = item["bbox"][0];
      const y = item["bbox"][1];
      const width = item["bbox"][2];
      const height = item["bbox"][3];

      // Draw the bounding box.
      ctx.strokeStyle = "#00FFFF";
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, width, height);
    });
  };

  render() {
    return (
      <div>
        <h1>Terracor Real-Time Diamond Count</h1>
        <h5>Count: {this.state.count} </h5>
        <video
          style={{ height: "500px", width: "500px" }}
          className="size"
          autoPlay
          playsInline
          muted
          ref={this.videoRef}
          width="500"
          height="500"
          id="frame"
        />
        <canvas
          className="size"
          ref={this.canvasRef}
          width="500"
          height="500"
        />
      </div>
    );
  }
}

const rootElement = document.getElementById("root");
ReactDOM.render(<App />, rootElement);
