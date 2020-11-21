import React from "react";
import Button from '@material-ui/core/Button';
import * as handpose from "@tensorflow-models/handpose";
import * as posenet from "@tensorflow-models/posenet";
import * as facemesh from "@tensorflow-models/facemesh";
//import * as tf from '@tensorflow/tfjs-core';
import Webcam from "react-webcam";
import "./App.css";
import { drawHand_tap, drawHand_rotate, drawHand_fist, drawHand_still, writeText, drawKeypoints, drawSkeleton, drawMesh } from "./utilities";
import { sqrt, pow } from "mathjs"
import {Line} from 'react-chartjs-2';

class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      ID : 0,
      tap_count : [],
      rotate_count : [],
      fist_count : [],
      index_passed : 0,
      min_dist: 0,
      max_dist: 5.0,
      rotate_passed : 0,
      last_pressed:0,
      real_time_inferencing:false,
      recording:false,
      button_mode:false,
      chart_ready:false,
      finger_done : false,
      rotate_done : false,
      fist_done : false,
      hand_dist_array : [],
      hand_rotate_array : [],
      hand_fist_array : [],
      hand_still_array : [],
      dist_array : [],
      dist_time_array : [],
      dist_record : [],
      dist_time_record : [],
      rotate_array : [],
      rotate_time_array : [],
      rotate_record : [],
      rotate_time_record : [],
      fist_array : [],
      fist_time_array : [],
      fist_record : [],
      fist_time_record : [],
      still_array : [],
      still_time_array : [],
      still_record : [],
      still_time_record : [],
      last_hand: [],
      chart_data1 : null,
      chart_data2 : null,
      chart_data3 : null,
      chart_data4 : null,
      wait : false,
      wait_till : 0,
      startAt: Date.now(),
      dead_frame: 0,
      raw: true,
      facingMode: "user",
      avg_fps: 0,
    };
    this.webcamRef = React.createRef(null);
    this.canvasRef = React.createRef(null);
    this.videoConstraints = {facingMode: "user"};
    this.capture_interval = 100;
    this.runHandpose = this.runHandpose.bind(this);
    this.stop_real_time_inference = this.stop_real_time_inference.bind(this);
    this.stop_tapping = this.stop_tapping.bind(this);
    this.stop_rotating = this.stop_rotating.bind(this);
    this.stop_gripping = this.stop_gripping.bind(this);
    this.stop_record = this.stop_record.bind(this);
    this.record_video = this.record_video.bind(this);
    this.concat_frame = this.concat_frame.bind(this);
    this.inference = this.inference.bind(this);
    this.reset_counter = this.reset_counter.bind(this);
    this.increment_tap1 = this.increment_tap1.bind(this);
    this.increment_tap2 = this.increment_tap2.bind(this);
    this.switch_button = this.switch_button.bind(this);
    this.compose_chart = this.compose_chart.bind(this);
    this.runPosenet = this.runPosenet.bind(this);
    this.real_time_posenet = this.real_time_posenet.bind(this);
    this.runFacemesh = this.runFacemesh.bind(this);
    this.real_time_facemesh = this.real_time_facemesh.bind(this);
    this.switch_style = this.switch_style.bind(this);
    this.switch_cam = this.switch_cam.bind(this);
    this.getArray = this.getArray.bind(this);
    this.exportToJson = this.exportToJson.bind(this);
  }

  switch_cam(){
    if (this.state.facingMode === "user") this.setState({facingMode: "environment"});
    else if (this.state.facingMode === "environment") this.setState({facingMode: "user"});
  }

  async runFacemesh(){
    require('@tensorflow/tfjs-backend-webgl');
    const net = await facemesh.load();
    console.log("Facemesh model loaded.");
    this.setState({startAt:Date.now()});
    const Interval_ID = setInterval(() => {
      this.real_time_facemesh(net);
    }, this.capture_interval);
    this.setState({ID:Interval_ID,
      real_time_inferencing:true});
  }

  async real_time_facemesh(net) {
    if (this.state.wait){
      this.setState({wait_till:Date.now()+3000});
      this.setState({wait:false});
    }
    if (
      typeof this.webcamRef.current !== "undefined" &&
      this.webcamRef.current !== null &&
      this.webcamRef.current.video.readyState === 4
    ) {
      const video = this.webcamRef.current.video;
      const videoWidth = this.webcamRef.current.video.videoWidth;
      const videoHeight = this.webcamRef.current.video.videoHeight;
      this.webcamRef.current.video.width = videoWidth;
      this.webcamRef.current.video.height = videoHeight;
      this.canvasRef.current.width = videoWidth;
      this.canvasRef.current.height = videoHeight;
      const face = await net.estimateFaces(video);
      const ctx = this.canvasRef.current.getContext("2d");
      drawMesh(face, ctx);
    }
  }

  async runPosenet(){
    require('@tensorflow/tfjs-backend-webgl');
    const net = await posenet.load();
    console.log("PoseNet model loaded.");
    this.setState({startAt:Date.now()});
    const Interval_ID = setInterval(() => {
      this.real_time_posenet(net);
    }, this.capture_interval);
    this.setState({ID:Interval_ID,
      real_time_inferencing:true});
  }

  async real_time_posenet(net) {
    if (this.state.wait){
      this.setState({wait_till:Date.now()+3000,
        wait:false});
    }
    if (
      typeof this.webcamRef.current !== "undefined" &&
      this.webcamRef.current !== null &&
      this.webcamRef.current.video.readyState === 4
    ) {
      const video = this.webcamRef.current.video;
      const videoWidth = this.webcamRef.current.video.videoWidth;
      const videoHeight = this.webcamRef.current.video.videoHeight;
      this.webcamRef.current.video.width = videoWidth;
      this.webcamRef.current.video.height = videoHeight;
      this.canvasRef.current.width = videoWidth;
      this.canvasRef.current.height = videoHeight;
      const pose = await net.estimateSinglePose(video);
      const ctx = this.canvasRef.current.getContext("2d");
      drawKeypoints(pose["keypoints"], 0.6, ctx);
      drawSkeleton(pose["keypoints"], 0.7, ctx);
    }
  }

  async switch_style(){
    await this.setState({raw: !this.state.raw});
    this.compose_chart();
  }

  compose_chart(){
    let time_array_1 = [];
    let count_array_1 = [];
    let label_1 = "";
    let time_array_2 = [];
    let count_array_2 = [];
    let label_2 = "";
    let time_array_3 = [];
    let count_array_3 = [];
    let label_3 = "";
    let time_array_4 = [];
    let count_array_4 = [];
    let label_4 = "";
    let real_dist = document.getElementById("real_measurement").value;
    let avg_fps = this.state.dist_time_array.length / 
                 (this.state.dist_time_array[this.state.dist_time_array.length - 1] - 
                  this.state.dist_time_array[0]);
    this.setState({avg_fps:avg_fps});
    
    if (this.state.raw){
      // Print Data Raw
      time_array_1 = [...this.state.dist_time_array];
      count_array_1 = [...this.state.dist_array];
      time_array_2 = [...this.state.rotate_time_array];
      count_array_2 = [...this.state.rotate_array];
      time_array_3 = [...this.state.fist_time_array];
      count_array_3 = [...this.state.fist_array];
      if (this.state.still_array.length > 100){
        time_array_4 = this.state.still_time_array.slice(0, this.state.still_time_array.length-10);
        count_array_4 = this.state.still_array.slice(0, this.state.still_array.length-10);
      }
      else {
        time_array_4 = [...this.state.still_time_array];
        count_array_4 = [...this.state.still_array];
      }
      
      
      // Change to Real Life Measurement
      if (real_dist > 0.0001){
        label_1 = "Distance between Index and Thumb (cm)";
        label_2 = "Relative Location between Left and Right of Hand (cm)";
        label_3 = "Relative Location between Tip of Fingers and Finger Joints (cm)";
        label_4 = "Relative Location Moved (cm)";
        let i = 0;
        let tmp = 0;
        for (i = 0; i < count_array_1.length; i++){
          tmp = count_array_1[i];
          count_array_1[i] = tmp*real_dist;
        }
        for (i = 0; i < count_array_2.length; i++){
          tmp = count_array_2[i];
          count_array_2[i] = tmp*real_dist;
        }
        for (i = 0; i < count_array_3.length; i++){
          tmp = count_array_3[i];
          count_array_3[i] = tmp*real_dist;
        }
        for (i = 0; i < count_array_4.length; i++){
          tmp = count_array_4[i];
          count_array_4[i] = tmp*real_dist;
        }
      }
      else{
        label_1 = "Distance between Index and Thumb (Relative Scale)";
        label_2 = "Relative Location between Left and Right of Hand (Relative Scale)";
        label_3 = "Relative Location between Tip of Fingers and Finger Joints (Relative Scale)";
        label_4 = "Relative Location Moved (Relative Scale)";
      }
    }
    else{
      // Recalculate Tapping Data
      let d_array = [];
      let avg_value = 0;
      for (let i = 1; i < this.state.dist_array.length; i++) d_array = [...d_array, Math.abs(this.state.dist_array[i-1] - this.state.dist_array[i])];
      for (let i = 0; i < d_array.length - 10; i++){
        avg_value = (d_array[i] + d_array[i+1] + d_array[i+2] + d_array[i+3] + d_array[i+4] + d_array[i+5] + d_array[i+6] + d_array[i+7] + d_array[i+8] + d_array[i+9])/10;
        count_array_1 = [...count_array_1, avg_value];
        time_array_1 = [...time_array_1, this.state.dist_time_array[i]];
      }

      d_array = [];
      avg_value = 0;
      for (let i = 1; i < this.state.rotate_array.length; i++) d_array = [...d_array, Math.abs(this.state.rotate_array[i-1] - this.state.rotate_array[i])];
      for (let i = 0; i < d_array.length - 10; i++){
        avg_value = (d_array[i] + d_array[i+1] + d_array[i+2] + d_array[i+3] + d_array[i+4] + d_array[i+5] + d_array[i+6] + d_array[i+7] + d_array[i+8] + d_array[i+9])/10;
        count_array_2 = [...count_array_2, avg_value];
        time_array_2 = [...time_array_2, this.state.rotate_time_array[i]];
      }

      d_array = [];
      avg_value = 0;
      for (let i = 1; i < this.state.fist_array.length; i++) d_array = [...d_array, Math.abs(this.state.fist_array[i-1] - this.state.fist_array[i])];
      for (let i = 0; i < d_array.length - 10; i++){
        avg_value = (d_array[i] + d_array[i+1] + d_array[i+2] + d_array[i+3] + d_array[i+4] + d_array[i+5] + d_array[i+6] + d_array[i+7] + d_array[i+8] + d_array[i+9])/10;
        count_array_3 = [...count_array_3, avg_value];
        time_array_3 = [...time_array_3, this.state.fist_time_array[i]];
      }

      d_array = [];
      avg_value = 0;
      for (let i = 1; i < this.state.still_array.length; i++) d_array = [...d_array, Math.abs(this.state.still_array[i-1] - this.state.still_array[i])];
      for (let i = 0; i < d_array.length - 10; i++){
        avg_value = (d_array[i] + d_array[i+1] + d_array[i+2] + d_array[i+3] + d_array[i+4] + d_array[i+5] + d_array[i+6] + d_array[i+7] + d_array[i+8] + d_array[i+9])/10;
        count_array_4 = [...count_array_4, avg_value];
        time_array_4 = [...time_array_4, this.state.fist_time_array[i]];
      }

      // Change to Real Life Measurement
      if (real_dist > 0.0001){
        label_1 = "Average Distance between Index and Thumb per Second (cm)";
        label_2 = "Average Distance between Left and Right of Hand per Second (cm)";
        label_3 = "Average Distance Location between Tip of Fingers and Finger Joints per Second (cm)";
        label_4 = "Relative Location Moved per Second (cm)";
        let i = 0;
        let tmp = 0;
        for (i = 0; i < count_array_1.length; i++){
          tmp = count_array_1[i];
          count_array_1[i] = tmp*real_dist;
        }
        for (i = 0; i < count_array_2.length; i++){
          tmp = count_array_2[i];
          count_array_2[i] = tmp*real_dist;
        }
        for (i = 0; i < count_array_3.length; i++){
          tmp = count_array_3[i];
          count_array_3[i] = tmp*real_dist;
        }
      }
      else{
        label_1 = "Average Distance between Index and Thumb per Second (Relative Scale)";
        label_2 = "Average Distance between Left and Right of Hand per Second (Relative Scale)";
        label_3 = "Average Distance Location between Tip of Fingers and Finger Joints per Second (Relative Scale)";
        label_4 = "Relative Location Moved per Second (Relative Scale)";
      }
    }

    // Setup Graphs
    const data1 = {
      labels: time_array_1,
      datasets: [        
        {
          label: label_1,
          fill: false,
          lineTension: 0.3,
          backgroundColor: 'rgba(75,192,192,0.4)',
          borderColor: 'rgba(75,192,192,1)',
          borderCapStyle: 'butt',
          borderDash: [],
          borderDashOffset: 0.0,
          borderJoinStyle: 'miter',
          pointBorderColor: 'rgba(75,192,192,1)',
          pointBackgroundColor: '#fff',
          pointBorderWidth: 1,
          pointHoverRadius: 5,
          pointHoverBackgroundColor: 'rgba(75,192,192,1)',
          pointHoverBorderColor: 'rgba(220,220,220,1)',
          pointHoverBorderWidth: 2,
          pointRadius: 1,
          pointHitRadius: 10,
          data: count_array_1
        }
      ]
    };
    const data2 = {
      labels: time_array_2,
      datasets: [        
        {
          label: label_2,
          fill: false,
          lineTension: 0.1,
          backgroundColor: 'rgba(192,75,192,0.4)',
          borderColor: 'rgba(192,75,192,1)',
          borderCapStyle: 'butt',
          borderDash: [],
          borderDashOffset: 0.0,
          borderJoinStyle: 'miter',
          pointBorderColor: 'rgba(192,75,192,1)',
          pointBackgroundColor: '#fff',
          pointBorderWidth: 1,
          pointHoverRadius: 5,
          pointHoverBackgroundColor: 'rgba(192,75,192,1)',
          pointHoverBorderColor: 'rgba(220,220,220,1)',
          pointHoverBorderWidth: 2,
          pointRadius: 1,
          pointHitRadius: 10,
          data: count_array_2 
        },
      ]
    };
    const data3 = {
      labels: time_array_3,
      datasets: [        
        {
          label: label_3,
          fill: false,
          lineTension: 0.1,
          backgroundColor: 'rgba(192,192,75,0.4)',
          borderColor: 'rgba(192,192,75,1)',
          borderCapStyle: 'butt',
          borderDash: [],
          borderDashOffset: 0.0,
          borderJoinStyle: 'miter',
          pointBorderColor: 'rgba(192,192,75,1)',
          pointBackgroundColor: '#fff',
          pointBorderWidth: 1,
          pointHoverRadius: 5,
          pointHoverBackgroundColor: 'rgba(192,192,75,1)',
          pointHoverBorderColor: 'rgba(220,220,220,1)',
          pointHoverBorderWidth: 2,
          pointRadius: 1,
          pointHitRadius: 10,
          data: count_array_3
        }
      ]
    };
    const data4 = {
      labels: time_array_4,
      datasets: [        
        {
          label: label_4,
          fill: false,
          lineTension: 0.1,
          backgroundColor: 'rgba(75,192,75,0.4)',
          borderColor: 'rgba(75,192,75,1)',
          borderCapStyle: 'butt',
          borderDash: [],
          borderDashOffset: 0.0,
          borderJoinStyle: 'miter',
          pointBorderColor: 'rgba(75,192,75,1)',
          pointBackgroundColor: '#fff',
          pointBorderWidth: 1,
          pointHoverRadius: 5,
          pointHoverBackgroundColor: 'rgba(75,192,75,1)',
          pointHoverBorderColor: 'rgba(220,220,220,1)',
          pointHoverBorderWidth: 2,
          pointRadius: 1,
          pointHitRadius: 10,
          data: count_array_4
        }
      ]
    };
    this.setState({chart_data1:data1});
    this.setState({chart_data2:data2});
    this.setState({chart_data3:data3});
    this.setState({chart_data4:data4});
    this.setState({chart_ready:true});
  }

  switch_button(){
    if (this.state.button_mode === true){
      this.setState({button_mode:false});
      this.compose_chart();
    }
    else{
      this.setState({button_mode:true})
    }
  }

  increment_tap1(){
    if (this.state.last_pressed !== 1){
      let current_moment = (Date.now() - this.state.startAt)/1000
      this.setState({last_pressed:1})
      this.setState({dist_array:[...this.state.dist_array, 1.0]});
      this.setState({dist_time_array:[...this.state.dist_time_array, current_moment]});
      this.setState({tap_count:[...this.state.tap_count, current_moment]});
    }
  }

  increment_tap2(){
    if (this.state.last_pressed !== 2){
      let current_moment = (Date.now() - this.state.startAt)/1000
      this.setState({last_pressed:2})
      this.setState({dist_array:[...this.state.dist_array, 1.0]});
      this.setState({dist_time_array:[...this.state.dist_time_array, current_moment]});
      this.setState({tap_count:[...this.state.tap_count, current_moment]});
    }
  }

  norm(lm1, lm2){
    return sqrt(pow(lm1[0]-lm2[0], 2)+pow(lm1[1]-lm2[1], 2))
  }

  reset_counter(){
    clearInterval(this.state.ID);
    const ctx = this.canvasRef.current.getContext("2d");
    ctx.clearRect(0,0, this.canvasRef.current.width, this.canvasRef.current.height);
    this.setState({
      ID : 0,
      tap_count : [],
      rotate_count : [],
      fist_count : [],
      index_passed : 0,
      min_dist: 0,
      max_dist: 5.0,
      rotate_passed : 0,
      last_pressed:0,
      real_time_inferencing:false,
      recording:false,
      button_mode:false,
      chart_ready:false,
      finger_done : false,
      rotate_done : false,
      fist_done : false,
      hand_dist_array : [],
      hand_rotate_array : [],
      hand_fist_array : [],
      hand_still_array : [],
      dist_array : [],
      dist_time_array : [],
      dist_record : [],
      dist_time_record : [],
      rotate_array : [],
      rotate_time_array : [],
      rotate_record : [],
      rotate_time_record : [],
      fist_array : [],
      fist_time_array : [],
      fist_record : [],
      fist_time_record : [],
      still_array : [],
      still_time_array : [],
      still_record : [],
      still_time_record : [],
      chart_data1 : null,
      chart_data2 : null,
      chart_data3 : null,
      wait : false,
      wait_till : 0,
      startAt: Date.now(),
      dead_frame: 0,
      raw: true,
      facingMode: "user",
      avg_fps: 0,
    });
  }

  sleep = (milliseconds) => {
    return new Promise(resolve => setTimeout(resolve, milliseconds))
  }

  async runHandpose() {
    const ctx = this.canvasRef.current.getContext("2d");
    writeText(ctx, { text: 'Loading', x: 180, y: 70 });
    require('@tensorflow/tfjs-backend-webgl');
    const net = await handpose.load();
    ctx.clearRect(0,0, this.canvasRef.current.width, this.canvasRef.current.height);
    //const uploadJSONInput = document.getElementById('upload-json');
    //const uploadWeightsInput = document.getElementById('upload-weights');
    //const model = await tf.loadLayersModel(tf.io.browserFiles([uploadJSONInput.files[0], uploadWeightsInput.files[0]]));
    console.log("Handpose model loaded.");
    this.setState({startAt:Date.now()});
    const Interval_ID = setInterval(() => {
      this.real_time_inference(net);
    }, this.capture_interval);
    this.setState({ID:Interval_ID,
      real_time_inferencing:true,
      wait:true});
  };

  async real_time_inference(net) {
    if (this.state.wait){
      this.setState({wait_till:Date.now()+3000});
      this.setState({wait:false});
    }
    if (
      typeof this.webcamRef.current !== "undefined" &&
      this.webcamRef.current !== null &&
      this.webcamRef.current.video.readyState === 4
    ) {
      const video = this.webcamRef.current.video;
      const videoWidth = this.webcamRef.current.video.videoWidth;
      const videoHeight = this.webcamRef.current.video.videoHeight;
      this.webcamRef.current.video.width = videoWidth;
      this.webcamRef.current.video.height = videoHeight;
      this.canvasRef.current.width = videoWidth;
      this.canvasRef.current.height = videoHeight;
      const hand = await net.estimateHands(video);
      const ctx = this.canvasRef.current.getContext("2d");
      
      //check if waiting
      if (Date.now() < this.state.wait_till){
        if (hand.length > 0) drawHand_tap (hand, ctx);
        //count down 3, 2, 1
        if (this.state.wait_till - Date.now() < 1000) writeText(ctx, { text: '1', x: 180, y: 70 });
        else if (this.state.wait_till - Date.now() < 2000) writeText(ctx, { text: '2', x: 140, y: 70 });
        else if (this.state.wait_till - Date.now() < 3000) writeText(ctx, { text: '3', x: 100, y: 70 });
      }
      else {
        if (hand.length > 0){
          hand.forEach((prediction) => {
            this.setState({dead_frame: 0});
            const landmarks = prediction.landmarks

            let pawn_dist = this.norm(landmarks[0], landmarks[2]);
            let current_moment = (Date.now() - this.state.startAt)/1000

            if (this.state.finger_done === false){
              drawHand_tap (hand, ctx);
              // Calculate relative distance
              let index_dist = this.norm(landmarks[4], landmarks[8]);
              let current_dist = index_dist/pawn_dist;

              // Record distance
              this.setState({dist_array:[...this.state.dist_array, current_dist],
                dist_time_array:[...this.state.dist_time_array, current_moment],
                hand_dist_array:[...this.state.hand_dist_array, hand]});
              
              // Perform counting
              if (this.state.index_passed === 0 && (current_dist - this.state.min_dist) > 0.05){
                this.setState({index_passed:1,
                  max_dist: current_dist});
                console.log(1);
              }
              if (this.state.index_passed === 1 && current_dist > this.state.max_dist){
                this.setState({max_dist: current_dist});
                console.log(2);
              }
              if (this.state.index_passed === 1 && (this.state.max_dist - current_dist) > 0.05){
                this.setState({index_passed:0,
                  min_dist: current_dist,
                  tap_count:[...this.state.tap_count, current_moment]});
                console.log(3);
              }
              if (this.state.index_passed === 0 && current_dist < this.state.min_dist){
                this.setState({min_dist: current_dist});
                console.log(4);
              }
            }
            
            if (this.state.finger_done === true && this.state.rotate_done === false){
              drawHand_rotate (hand, ctx, this.state.rotate_passed);
              // Calculate relative distance
              let rotate_dist = (landmarks[2][0] - landmarks[17][0]) / pawn_dist;

              // Record distance
              this.setState({rotate_array:[...this.state.rotate_array, rotate_dist],
                rotate_time_array:[...this.state.rotate_time_array, current_moment],
                hand_rotate_array:[...this.state.hand_rotate_array, hand]});

              // Perform counting
              if (this.state.rotate_passed === 0){
                if (rotate_dist >= 0.25) this.setState({rotate_passed:1});
                if (rotate_dist <= -0.25) this.setState({rotate_passed:-1});
              }
              if (this.state.rotate_passed === 1 && rotate_dist <= -0.5){
                this.setState({rotate_passed:-1});
              }
              if (this.state.rotate_passed === -1 && rotate_dist >= 0.5){
                this.setState({rotate_passed:1,
                  rotate_count:[...this.state.rotate_count, current_moment]});
              }
            }

            if (this.state.rotate_done === true && this.state.fist_done === false){
              console.log(this.state.fist_passed);
              drawHand_fist (hand, ctx, this.state.fist_passed);
              // Calculate relative distance
              let fist_dist =  ((landmarks[8][1] - landmarks[5][1])+
                                (landmarks[12][1] - landmarks[9][1])+
                                (landmarks[16][1] - landmarks[13][1])+
                                (landmarks[20][1] - landmarks[17][1]))/
                                (4*pawn_dist)

              // Record distance
              this.setState({fist_array:[...this.state.fist_array, fist_dist],
                fist_time_array:[...this.state.fist_time_array, current_moment],
                hand_fist_array:[...this.state.hand_fist_array, hand]});

              // Perform counting
              if (fist_dist >= 0.0){this.setState({fist_passed:1})}
              if (fist_dist < -0.4 && this.state.fist_passed === 1){
                this.setState({fist_passed:0,
                  fist_count:[...this.state.fist_count, current_moment]});
              }
            }
            
            if (this.state.fist_done === true ){
              // Calculate relative distance
              let total_move;
              if (this.state.last_hand.length > 0){
                let i;
                let moved = [];
                total_move = 0.0;
                for (i = 0; i < 21; i++){
                  let move_dist = this.norm(landmarks[i], this.state.last_hand[i]) / pawn_dist;
                  total_move += move_dist;
                  if ( move_dist > 0.05){
                    moved = [...moved, i];
                  }
                }
                console.log(moved);
                drawHand_still (hand, ctx, moved);
              }

              // Record Hand Landmarks
              this.setState({still_array:[...this.state.still_array, total_move],
                still_time_array:[...this.state.still_time_array, current_moment],
                hand_still_array:[...this.state.hand_still_array, hand],
                last_hand: landmarks,
              });
            }

          });
        }
        else {
          // Warning message for hand off screen
          if (this.state.dead_frame > 9) writeText(ctx, { text: 'Hand Off Screen', x: 180, y: 70 });
          this.setState({dead_frame: this.state.dead_frame + 1});
        }
      }
    }
  };

  async stop_tapping() {
    this.setState({finger_done:true});
    this.setState({wait:true});
  }

  async stop_rotating() {
    this.setState({rotate_done:true});
    this.setState({wait:true});
  }

  async stop_gripping() {
    this.setState({fist_done:true});
    this.setState({wait:true});
  }

  stop_real_time_inference() {
    clearInterval(this.state.ID);
    this.setState({real_time_inferencing:false,
                   finger_done:false,
                   rotate_done:false,
                   fist_done:false});
    this.compose_chart();
    const ctx = this.canvasRef.current.getContext("2d");
    ctx.clearRect(0,0, this.canvasRef.current.width, this.canvasRef.current.height);
  }

  async record_video(){
    this.setState({startAt:Date.now()});
    console.log("Handpose model loaded.");
    const Interval_ID = setInterval(() => {
      this.concat_frame();
    }, this.capture_interval);
    this.setState({ID:Interval_ID,
      recording:true,
      wait:true});
  }

  async concat_frame() {
    if (this.state.wait){
      this.setState({wait_till:Date.now()+3000,
        wait:false});
    }
    if (
      typeof this.webcamRef.current !== "undefined" &&
      this.webcamRef.current !== null &&
      this.webcamRef.current.video.readyState === 4
    ) {
      if (Date.now() < this.state.wait_till){
        console.log("Waiting till ", this.state.wait_till);
        //count down 3, 2, 1
        const ctx = this.canvasRef.current.getContext("2d");
        if (this.state.wait_till - Date.now() < 250) ctx.clearRect(0,0, this.canvasRef.current.width, this.canvasRef.current.height);
        else if (this.state.wait_till - Date.now() < 1000) writeText(ctx, { text: '1', x: 180, y: 70 });
        else if (this.state.wait_till - Date.now() < 2000) writeText(ctx, { text: '2', x: 140, y: 70 });
        else if (this.state.wait_till - Date.now() < 3000) writeText(ctx, { text: '3', x: 100, y: 70 });
      }
      else {
        let current_moment = (Date.now() - this.state.startAt)/1000;
        const image = this.webcamRef.current.getScreenshot();      
        var img = document.createElement("img");
        img.src = image;
        img.onload = function(){
          if (this.state.finger_done === false){
            this.setState({dist_record:[...this.state.dist_record, img],
              dist_time_record:[...this.state.dist_time_record, current_moment]});
          }

          else if (this.state.finger_done === true && this.state.rotate_done === false){
            this.setState({rotate_record:[...this.state.rotate_record, img],
              rotate_time_record:[...this.state.rotate_time_record, current_moment]});
          }

          else if (this.state.rotate_done === true && this.state.fist_done === false){
            this.setState({fist_record:[...this.state.fist_record, img],
              fist_time_record:[...this.state.fist_time_record, current_moment]});
          }

          if (this.state.fist_done === true ){
            this.setState({still_record:[...this.state.still_record, img],
              still_time_record:[...this.state.still_time_record, current_moment]});
          }
        }.bind(this)
      }
    }
  }

  async inference() {
    require('@tensorflow/tfjs-backend-webgl');
    const net = await handpose.load();
    console.log("Handpose model loaded.");

    // Run prediction on recorded tapping data
    for (let i = 0; i<this.state.dist_record.length; i++){
      const hand = await net.estimateHands(this.state.dist_record[i]);
      if (hand.length > 0){
        hand.forEach((prediction) => { 
          // Calculate relative distance
          const landmarks = prediction.landmarks
          let index_dist = this.norm(landmarks[4], landmarks[8]);
          let pawn_dist = this.norm(landmarks[0], landmarks[2]);
          let current_dist = index_dist/pawn_dist;

          // Record distance
          this.setState({dist_array:[...this.state.dist_array, current_dist],
            dist_time_array:[...this.state.dist_time_array, this.state.dist_time_record[i]],
            hand_dist_array:[...this.state.hand_dist_array, hand]});
          
          // Perform counting
          if (this.state.index_passed === 0 && (current_dist - this.state.min_dist) > 0.05){
            this.setState({index_passed:1,
              max_dist: current_dist});
            console.log(1);
          }
          if (this.state.index_passed === 1 && current_dist > this.state.max_dist){
            this.setState({max_dist: current_dist});
            console.log(2);
          }
          if (this.state.index_passed === 1 && (this.state.max_dist - current_dist) > 0.05){
            this.setState({index_passed:0,
              min_dist: current_dist,
              tap_count:[...this.state.tap_count, this.state.dist_time_record[i]]});
            console.log(3);
          }
          if (this.state.index_passed === 0 && current_dist < this.state.min_dist){
            this.setState({min_dist: current_dist});
            console.log(4);
          }
          console.log("INDEX COUNT:", this.state.tap_count);
        });
      }
    }

    // Run prediction on recorded rotation data
    for (let i = 0; i<this.state.rotate_record.length; i++){
      const hand = await net.estimateHands(this.state.rotate_record[i]);
      if (hand.length > 0){
        hand.forEach((prediction) => {
          const landmarks = prediction.landmarks
          let pawn_dist = this.norm(landmarks[0], landmarks[2]);
          let rotate_dist = (landmarks[2][0] - landmarks[17][0]) / pawn_dist;
          //this.setState({pawn_rotate_array:[...this.state.pawn_rotate_array, pawn_dist]});
          this.setState({rotate_array:[...this.state.rotate_array, rotate_dist],
            rotate_time_array:[...this.state.rotate_time_array, this.state.rotate_time_record[i]],
            hand_rotate_array:[...this.state.hand_rotate_array, hand]});
          if (this.state.rotate_passed === 0){
            if (rotate_dist >= 0.5) this.setState({rotate_passed:1});
            if (rotate_dist <= -0.5) this.setState({rotate_passed:-1});
          }
          if (this.state.rotate_passed === 1 && rotate_dist <= -0.5){
            this.setState({rotate_passed:-1});
          }
          if (this.state.rotate_passed === -1 && rotate_dist >= 0.5){
            this.setState({rotate_passed:1, 
              rotate_count:[...this.state.rotate_count, this.state.rotate_time_record[i]]});
          }
          console.log("ROTATE COUNT:", this.state.rotate_count);
        });
      }
    }

    // Run prediction on recorded gripping data
    for (let i = 0; i<this.state.fist_record.length; i++){
      const hand = await net.estimateHands(this.state.fist_record[i]);
      if (hand.length > 0){
        hand.forEach((prediction) => {
          const landmarks = prediction.landmarks
          let pawn_dist = this.norm(landmarks[0], landmarks[2]);
          let fist_dist =  ((landmarks[8][1] - landmarks[5][1])+
            (landmarks[12][1] - landmarks[9][1])+
            (landmarks[16][1] - landmarks[13][1])+
            (landmarks[20][1] - landmarks[17][1]))/
            (4*pawn_dist)
          //this.setState({pawn_fist_array:[...this.state.pawn_fist_array, pawn_dist]});
          this.setState({fist_array:[...this.state.fist_array, fist_dist],
            fist_time_array:[...this.state.fist_time_array, this.state.fist_time_record[i]],
            hand_fist_array:[...this.state.hand_fist_array, hand]});
          if (fist_dist >= 0.0){this.setState({fist_passed:1})}
          if (fist_dist < -0.4 && this.state.fist_passed === 1){
            this.setState({fist_passed:0});
            this.setState({fist_count:[...this.state.fist_count, this.state.fist_time_record[i]]});
          }
          console.log("FIST COUNT:", this.state.fist_count);
        });
      }
    }

    // Run prediction on recorded postural data
    for (let i = 0; i<this.state.still_record.length; i++){
      const hand = await net.estimateHands(this.state.still_record[i]);
      if (hand.length > 0){
        hand.forEach((prediction) => {
          const landmarks = prediction.landmarks
          let pawn_dist = this.norm(landmarks[0], landmarks[2]);
          let total_move;
          let i;
          let moved = [];
          if (this.state.last_hand.length > 0){
            total_move = 0.0;
            for (i = 0; i < 21; i++){
              let move_dist = this.norm(landmarks[i], this.state.last_hand[i]) / pawn_dist;
              total_move += move_dist;
              if ( move_dist > 0.1){
                moved = [...moved, i];
              }
            }
          }
          console.log("TOTAL MOVED:", total_move);
          //this.setState({pawn_fist_array:[...this.state.pawn_fist_array, pawn_dist]});
          this.setState({still_array:[...this.state.still_array, total_move],
            still_time_array:[...this.state.still_time_array, this.state.still_time_record[i]],
            hand_still_array:[...this.state.hand_still_array, hand],
            last_hand: landmarks,
          });
        });
      }
    }

    this.setState({record:[]});
  }

  async stop_record() {
    clearInterval(this.state.ID);
    await this.inference();
    this.setState({recording:false,
      finger_done:false,
      rotate_done:false,
      fist_done:false});
    this.compose_chart();
  }

  exportToJson = (objectData, filename) => {
    let contentType = "application/json;charset=utf-8;";
    if (window.navigator && window.navigator.msSaveOrOpenBlob) {
      var blob = new Blob([decodeURIComponent(encodeURI(JSON.stringify(objectData)))], { type: contentType });
      navigator.msSaveOrOpenBlob(blob, filename);
    } else {
      var a = document.createElement('a');
      a.download = filename;
      a.href = 'data:' + contentType + ',' + encodeURIComponent(JSON.stringify(objectData));
      a.target = '_blank';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  }
 
  getArray = () => {
    let dict = {
      tap_count : this.state.tap_count,
      rotate_count : this.state.rotate_count,
      fist_count : this.state.fist_count,
      dist_array : this.state.dist_array,
      dist_time_array : this.state.dist_time_array,
      dist_record : this.state.dist_record,
      dist_time_record : this.state.dist_time_record,
      rotate_array : this.state.rotate_array,
      rotate_time_array : this.state.rotate_time_array,
      rotate_record : this.state.rotate_record,
      rotate_time_record : this.state.rotate_time_record,
      fist_array : this.state.fist_array,
      fist_time_array : this.state.fist_time_array,
      fist_record : this.state.fist_record,
      fist_time_record : this.state.fist_time_record,
      still_array : this.state.still_array,
      still_time_array : this.state.still_time_array,
      still_record : this.state.still_record,
      still_time_record : this.state.still_time_record,
      startAt: this.state.startAt,
      avg_fps: this.state.avg_fps,
      hand_dist_array : this.state.hand_dist_array,
      hand_rotate_array : this.state.hand_rotate_array,
      hand_fist_array : this.state.hand_fist_array,
      hand_still_array : this.state.hand_still_array,
    }
    this.exportToJson(dict, "state");
  }

  render(){
    const videoConstraints = {
      facingMode: this.state.facingMode
    };
    return (
      <div className="App">
        <header className="App-header">
        <Webcam
          ref={this.webcamRef}
          videoConstraints={videoConstraints}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zindex: 9,
            width: "auto",
            height: "auto",
          }}
        />
        <canvas
          ref={this.canvasRef}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zindex: 9,
            width: "auto",
            height: "auto",
          }}
        />
        </header>
          {this.state.button_mode ? (
            <div>
              <button onClick={this.increment_tap1} id="button1">Index Finger</button>
              <button onClick={this.increment_tap2} id="button2">Middle Finger</button>
            </div>
          ) : (
            <div></div>
          )}

          <div>
            {this.state.real_time_inferencing ? (
              this.state.finger_done ? (
                this.state.rotate_done ? (
                  this.state.fist_done ? (
                    <Button variant="contained" color="primary"  onClick={this.stop_real_time_inference}>Calculate Result</Button>
                  ):(
                    <Button variant="contained" color="primary"  onClick={this.stop_gripping}>Finish Gripping</Button>
                  )                  
                ):(
                  <Button variant="contained" color="primary"  onClick={this.stop_rotating}>Finish Rotating</Button>
                )
              ):(
                <Button variant="contained" color="primary"  onClick={this.stop_tapping}>Finish Tapping</Button>
              )            
            ) : (
            <Button disabled={this.state.recording} variant="contained" color="primary"  onClick={this.runHandpose}>Starting Real Time Inference</Button>
            )}

            {this.state.recording ? (
              this.state.finger_done ? (
                this.state.rotate_done ? (
                  this.state.fist_done ? (
                    <Button variant="contained" color="secondary" onClick={this.stop_record}>Calculate Result</Button>
                  ) : (
                    <Button variant="contained" color="secondary"  onClick={this.stop_gripping}>Finish Gripping</Button>
                  )                  
                ):(
                  <Button variant="contained" color="secondary" onClick={this.stop_rotating}>Finish Rotating</Button>
                )
              ):(
                <Button variant="contained" color="secondary" onClick={this.stop_tapping}>Finish Tapping</Button>
              )            
            ) : (
            <Button disabled={this.state.real_time_inferencing} variant="contained" color="secondary" onClick={this.record_video}>Starting Recording</Button>
            )}

            <Button disabled={this.state.recording||this.state.real_time_inferencing} variant="outlined" color="secondary" onClick={this.reset_counter}>
              Reset All
            </Button>
          </div>
          <div>
            <h5>
              Finger Tapping Count:{this.state.tap_count.length}&nbsp;&nbsp;&nbsp;&nbsp;
              Rotate Count:{this.state.rotate_count.length}&nbsp;&nbsp;&nbsp;&nbsp;
              Gripping Count: {this.state.fist_count.length}&nbsp;&nbsp;&nbsp;&nbsp;
              FPS: {this.state.avg_fps}
            </h5>
          </div>
          <div>
            <small>Enter Real Measurement (cm)</small>
            <input type="number" id="real_measurement" onChange={this.compose_chart} step="0.001" min='0' max='20'></input>
            <button disabled={!this.state.chart_ready} onClick={this.switch_style}>Switch Chart Style</button>
            <button onClick={this.switch_cam}>Switch Camera</button>
          </div>
          <div>
            <small>Experimental Features</small>
            <button disabled={this.state.recording||this.state.real_time_inferencing} onClick={this.runPosenet}>PoseNet</button>
            <button disabled={this.state.recording||this.state.real_time_inferencing} onClick={this.runFacemesh}>Facemesh</button>
            <button disabled={this.state.recording||this.state.real_time_inferencing} onClick={this.switch_button}>Switch On/Off Button</button>
          </div>
          <div>
            {this.state.chart_ready ? (
              <div>
                <Line data={this.state.chart_data1} />
                <Line data={this.state.chart_data2} />
                <Line data={this.state.chart_data3} />
                <Line data={this.state.chart_data4} />
              </div>
            ) : (
              <div/>
            )}
            <div>
              <input type="file" id="upload-json"></input>
            </div>
            <div>
              <input type="file" id="upload-weights"></input>
            </div>
            <button onClick={this.getArray}>Get Array</button>
          </div>
      </div>
    );
  }
}

export default App;