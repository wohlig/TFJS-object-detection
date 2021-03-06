import React from "react";
import ReactDOM from "react-dom";
import * as tf from "@tensorflow/tfjs";
import _ from "lodash";
import moment from "moment";
import "./styles.css";
tf.setBackend("webgl");

const threshold = 0.4;
var startTime = moment();
var diamondcount = 0;

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

let classesDir = {
  1: {
    name: "diamond",
    id: 1,
  },
};

class App extends React.Component {
  videoRef = React.createRef();
  canvasRef = React.createRef();
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
      console.log(prediction);
      var boxes = prediction[2].dataSync();

      // console.log("boxesboxesboxes", boxes)

      boxes = _.map(boxes, function(value) {
        return parseInt(value * 320);
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
        // var temp=obj.score >= threshold
        return obj.score >= threshold;
      });
      console.log("TOTAL DETECTION:", finalResponse.length);
      // console.log("TOTAL DETECTION:",finalResponse.length)
      var i;
      for (i = 0; i < finalResponse.length; i++) {
        console.log(
          "Diamond " + (i + 1) + " with Score: " + finalResponse[i].score
        );
        console.log("Box Coordinates:" + finalResponse[i].box);
        console.log("------------------------------------");
      }

      console.log(
        "Response Time: ",
        moment().diff(startTime, "ms") / 1000,
        "Seconds"
      );

      // this.renderPredictions(prediction, video);
      requestAnimationFrame(() => {
        startTime = moment();
        this.detectFrame(video, model);
      });
      tf.engine().endScope();
    });
  };

  process_input(video_frame) {
    const tfimg = tf.browser.fromPixels(video_frame).toInt();
    const expandedimg = tfimg.transpose([0, 1, 2]).expandDims();
    return expandedimg;
  }

  buildDetectedObjects(scores, threshold, boxes, classes, classesDir) {
    const detectionObjects = [];
    var video_frame = document.getElementById("frame");

    scores[0].forEach((score, i) => {
      if (score > threshold) {
        const bbox = [];
        const minY = boxes[0][i][0] * video_frame.offsetHeight;
        const minX = boxes[0][i][1] * video_frame.offsetWidth;
        const maxY = boxes[0][i][2] * video_frame.offsetHeight;
        const maxX = boxes[0][i][3] * video_frame.offsetWidth;
        bbox[0] = minX;
        bbox[1] = minY;
        bbox[2] = maxX - minX;
        bbox[3] = maxY - minY;
        detectionObjects.push({
          class: classes[i],
          label: classesDir[classes[i]].name,
          score: score.toFixed(4),
          bbox: bbox,
        });
      }
    });
    return detectionObjects;
  }

  renderPredictions = (predictions) => {
    const ctx = this.canvasRef.current.getContext("2d");
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    // Font options.
    const font = "16px sans-serif";
    ctx.font = font;
    ctx.textBaseline = "top";

    //Getting predictions
    const boxes = predictions[4].arraySync();
    const scores = predictions[5].arraySync();
    const classes = predictions[6].dataSync();
    const detections = this.buildDetectedObjects(
      scores,
      threshold,
      boxes,
      classes,
      classesDir
    );

    detections.forEach((item) => {
      const x = item["bbox"][0];
      const y = item["bbox"][1];
      const width = item["bbox"][2];
      const height = item["bbox"][3];

      // Draw the bounding box.
      ctx.strokeStyle = "#00FFFF";
      ctx.lineWidth = 4;
      ctx.strokeRect(x, y, width, height);

      // Draw the label background.
      ctx.fillStyle = "#00FFFF";
      const textWidth = ctx.measureText(
        item["label"] + " " + (100 * item["score"]).toFixed(2) + "%"
      ).width;
      const textHeight = parseInt(font, 10); // base 10
      ctx.fillRect(x, y, textWidth + 4, textHeight + 4);
    });

    detections.forEach((item) => {
      const x = item["bbox"][0];
      const y = item["bbox"][1];

      // Draw the text last to ensure it's on top.
      ctx.fillStyle = "#000000";
      ctx.fillText(
        item["label"] + " " + (100 * item["score"]).toFixed(2) + "%",
        x,
        y
      );
    });
  };

  render() {
    return (
      <div>
        <h1>Terracor Real-Time Diamond Count</h1>
        <video
          style={{ height: "600px", width: "500px" }}
          className="size"
          autoPlay
          playsInline
          muted
          ref={this.videoRef}
          width="320"
          height="320"
          id="frame"
        />
        <canvas
          className="size"
          ref={this.canvasRef}
          width="320"
          height="320"
        />
      </div>
    );
  }
}

const rootElement = document.getElementById("root");
ReactDOM.render(<App />, rootElement);
