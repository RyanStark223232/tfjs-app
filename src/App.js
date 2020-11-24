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
import handline from './HandLines1.jpg';

class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      ID : 0,
      tap_count : [],
      rotate_count : [],
      fist_count : [],
      last_hand_L: [],
      last_hand_R: [],
      index_passed : 0,
      min_dist: 0,
      max_dist: 999.0,
      rotate_passed : 0,
      last_pressed:0,

      real_time_inferencing:false,
      recording:false,
      button_mode:false,
      chart_ready:false,
      finger_done : false,
      rotate_done : false,
      fist_done : false,

      hand_dist_array_L : [],
      hand_dist_array_R : [],
      hand_rotate_array_L : [],
      hand_rotate_array_R : [],
      hand_fist_array_L : [],
      hand_fist_array_R : [],
      hand_still_array_L : [],
      hand_still_array_R : [],

      dist_array_L : [],
      dist_time_array_L : [],
      dist_record_L : [],
      dist_time_record_L : [],
      dist_array_R : [],
      dist_time_array_R : [],
      dist_record_R : [],
      dist_time_record_R : [],

      rotate_array_L : [],
      rotate_time_array_L : [],
      rotate_record_L : [],
      rotate_time_record_L : [],
      rotate_array_R : [],
      rotate_time_array_R : [],
      rotate_record_R : [],
      rotate_time_record_R : [],

      fist_array_L : [],
      fist_time_array_L : [],
      fist_record_L : [],
      fist_time_record_L : [],
      fist_array_R : [],
      fist_time_array_R : [],
      fist_record_R : [],
      fist_time_record_R : [],

      still_array_L : [],
      still_time_array_L : [],
      still_record_L : [],
      still_time_record_L : [],
      still_array_R : [],
      still_time_array_R : [],
      still_record_R : [],
      still_time_record_R : [],

      chart_data1_L : null,
      chart_data1_R : null,
      chart_data2_L : null,
      chart_data2_R : null,
      chart_data3_L : null,
      chart_data3_R : null,
      chart_data4_L : null,
      chart_data4_R : null,

      stage:0,
      wait : false,
      wait_till : 0,
      max_tremor_L: 0.0,
      max_tremor_R: 0.0,

      startAt: Date.now(),
      dead_frame: 0,
      raw: true,
      facingMode: "user",
      avg_fps: 0,
    };
    this.webcamRef = React.createRef(null);
    this.canvasRef = React.createRef(null);
    this.videoConstraints = {facingMode: "user"};
    this.capture_interval = 50;
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
    this.next_step = this.next_step.bind(this);
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
    let time_array_1_L = [];
    let time_array_1_R = [];
    let count_array_1_L = [];
    let count_array_1_R = [];
    let label_1_L = "";
    let label_1_R = "";
    let time_array_2_L = [];
    let time_array_2_R = [];
    let count_array_2_L = [];
    let count_array_2_R = [];
    let label_2_L = "";
    let label_2_R = "";
    let time_array_3_L = [];
    let time_array_3_R = [];
    let count_array_3_L = [];
    let count_array_3_R = [];
    let label_3_L = "";
    let label_3_R = "";
    let time_array_4_L = [];
    let time_array_4_R = [];
    let count_array_4_L = [];
    let count_array_4_R = [];
    let label_4_L = "";
    let label_4_R = "";
    let real_dist_v = document.getElementById("real_measurement_v").value;
    let real_dist_h = document.getElementById("real_measurement_h").value;
    let avg_fps = this.state.hand_dist_array_L.length / 
                 (this.state.dist_time_array_L[this.state.dist_time_array_L.length - 1] - 
                  this.state.dist_time_array_L[0]);
    this.setState({avg_fps:avg_fps});
    
    if (this.state.raw){
      // Print Data Raw
      if (this.state.dist_array_L.length > 100){
        time_array_1_L = this.state.dist_time_array_L.slice(10, this.state.dist_time_array_L.length-20);
        count_array_1_L = this.state.dist_array_L.slice(10, this.state.dist_array_L.length-20);
      }
      else {
        time_array_1_L = [...this.state.dist_time_array_L];
        count_array_1_L = [...this.state.dist_array_L];
      }
      if (this.state.dist_array_R.length > 100){
        time_array_1_R = this.state.dist_time_array_R.slice(10, this.state.dist_time_array_R.length-20);
        count_array_1_R = this.state.dist_array_R.slice(10, this.state.dist_array_R.length-20);
      }
      else {
        time_array_1_R = [...this.state.dist_time_array_R];
        count_array_1_R = [...this.state.dist_array_R];
      }
      if (this.state.rotate_array_L.length > 100){
        time_array_2_L = this.state.rotate_time_array_L.slice(10, this.state.rotate_time_array_L.length-20);
        count_array_2_L = this.state.rotate_array_L.slice(10, this.state.rotate_array_L.length-20);
      }
      else {
        time_array_2_L = [...this.state.rotate_time_array_L];
        count_array_2_L = [...this.state.rotate_array_L];
      }
      if (this.state.rotate_array_R.length > 100){
        time_array_2_R = this.state.rotate_time_array_R.slice(10, this.state.rotate_time_array_R.length-20);
        count_array_2_R = this.state.rotate_array_R.slice(10, this.state.rotate_array_R.length-20);
      }
      else {
        time_array_2_R = [...this.state.rotate_time_array_R];
        count_array_2_R = [...this.state.rotate_array_R];
      }
      if (this.state.fist_array_L.length > 100){
        time_array_3_L = this.state.fist_time_array_L.slice(10, this.state.fist_time_array_L.length-20);
        count_array_3_L = this.state.fist_array_L.slice(10, this.state.fist_array_L.length-20);
      }
      else {
        time_array_3_L = [...this.state.fist_time_array_L];
        count_array_3_L = [...this.state.fist_array_L];
      }
      if (this.state.fist_array_R.length > 100){
        time_array_3_R = this.state.fist_time_array_R.slice(10, this.state.fist_time_array_R.length-20);
        count_array_3_R = this.state.fist_array_R.slice(10, this.state.fist_array_R.length-20);
      }
      else {
        time_array_3_R = [...this.state.fist_time_array_R];
        count_array_3_R = [...this.state.fist_array_R];
      }
      if (this.state.still_array_L.length > 100){
        time_array_4_L = this.state.still_time_array_L.slice(10, this.state.still_time_array_L.length-20);
        count_array_4_L = this.state.still_array_L.slice(10, this.state.still_array_L.length-20);
      }
      else {
        time_array_4_L = [...this.state.still_time_array_L];
        count_array_4_L = [...this.state.still_array_L];
      }
      if (this.state.still_array_R.length > 100){
        time_array_4_R = this.state.still_time_array_R.slice(10, this.state.still_time_array_R.length-20);
        count_array_4_R = this.state.still_array_R.slice(10, this.state.still_array_R.length-20);
      }
      else {
        time_array_4_R = [...this.state.still_time_array_R];
        count_array_4_R = [...this.state.still_array_R];
      }
      
      // Change to Real Life Measurement
      if (real_dist_v > 0.0001 && real_dist_h > 0.0001){
        label_1_L = "Distance between Index and Thumb (cm)";
        label_1_R = "Distance between Index and Thumb (cm)";
        label_2_L = "Relative Location between Left and Right of Hand (cm)";
        label_2_R = "Relative Location between Left and Right of Hand (cm)";
        label_3_L = "Relative Location between Tip of Fingers and Finger Joints (cm)";
        label_3_R = "Relative Location between Tip of Fingers and Finger Joints (cm)";
        label_4_L = "Relative Location Moved (cm)";
        label_4_R = "Relative Location Moved (cm)";
        let i = 0;
        let tmp = 0;
        for (i = 0; i < count_array_1_L.length; i++){
          tmp = count_array_1_L[i];
          count_array_1_L[i] = tmp*real_dist_h;
        }
        for (i = 0; i < count_array_1_R.length; i++){
          tmp = count_array_1_R[i];
          count_array_1_R[i] = tmp*real_dist_h;
        }
        for (i = 0; i < count_array_2_L.length; i++){
          tmp = count_array_2_L[i];
          count_array_2_L[i] = tmp*real_dist_v;
        }
        for (i = 0; i < count_array_2_R.length; i++){
          tmp = count_array_2_R[i];
          count_array_2_R[i] = tmp*real_dist_v;
        }
        for (i = 0; i < count_array_3_L.length; i++){
          tmp = count_array_3_L[i];
          count_array_3_L[i] = tmp*real_dist_h;
        }
        for (i = 0; i < count_array_3_R.length; i++){
          tmp = count_array_3_R[i];
          count_array_3_R[i] = tmp*real_dist_h;
        }
        for (i = 0; i < count_array_4_L.length; i++){
          tmp = count_array_4_L[i];
          count_array_4_L[i] = tmp*real_dist_h;
        }
        for (i = 0; i < count_array_4_R.length; i++){
          tmp = count_array_4_R[i];
          count_array_4_R[i] = tmp*real_dist_h;
        }
      }
      else{
        label_1_L = "Distance between Index and Thumb";
        label_1_R = "Distance between Index and Thumb";
        label_2_L = "Relative Location between Left and Right of Hand";
        label_2_R = "Relative Location between Left and Right of Hand";
        label_3_L = "Relative Location between Tip of Fingers and Finger Joints";
        label_3_R = "Relative Location between Tip of Fingers and Finger Joints";
        label_4_L = "Relative Location Moved";
        label_4_R = "Relative Location Moved";
      }
    }
    // Setup Graphs
    const data1_L = {
      labels: time_array_1_L,
      datasets: [        
        {
          label: label_1_L,
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
          data: count_array_1_L
        }
      ]
    };
    const data1_R = {
      labels: time_array_1_R,
      datasets: [        
        {
          label: label_1_R,
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
          data: count_array_1_R
        }
      ]
    };
    const data2_L = {
      labels: time_array_2_L,
      datasets: [        
        {
          label: label_2_L,
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
          data: count_array_2_L 
        },
      ]
    };
    const data2_R = {
      labels: time_array_2_R,
      datasets: [        
        {
          label: label_2_R,
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
          data: count_array_2_R 
        },
      ]
    };
    const data3_L = {
      labels: time_array_3_L,
      datasets: [        
        {
          label: label_3_L,
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
          data: count_array_3_L
        }
      ]
    };
    const data3_R = {
      labels: time_array_3_R,
      datasets: [        
        {
          label: label_3_R,
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
          data: count_array_3_R
        }
      ]
    };
    const data4_L = {
      labels: time_array_4_L,
      datasets: [        
        {
          label: label_4_L,
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
          data: count_array_4_L
        }
      ]
    };
    const data4_R = {
      labels: time_array_4_R,
      datasets: [        
        {
          label: label_4_R,
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
          data: count_array_4_R
        }
      ]
    };
    this.setState({chart_data1_L:data1_L});
    this.setState({chart_data1_R:data1_R});
    this.setState({chart_data2_L:data2_L});
    this.setState({chart_data2_R:data2_R});
    this.setState({chart_data3_L:data3_L});
    this.setState({chart_data3_R:data3_R});
    this.setState({chart_data4_L:data4_L});
    this.setState({chart_data4_R:data4_R});
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
      last_hand_L: [],
      last_hand_R: [],
      index_passed : 0,
      min_dist: 0,
      max_dist: 99.0,
      rotate_passed : 0,
      last_pressed:0,

      real_time_inferencing:false,
      recording:false,
      button_mode:false,
      chart_ready:false,
      finger_done : false,
      rotate_done : false,
      fist_done : false,

      hand_dist_array_L : [],
      hand_dist_array_R : [],
      hand_rotate_array_L : [],
      hand_rotate_array_R : [],
      hand_fist_array_L : [],
      hand_fist_array_R : [],
      hand_still_array_L : [],
      hand_still_array_R : [],

      dist_array_L : [],
      dist_time_array_L : [],
      dist_record_L : [],
      dist_time_record_L : [],
      dist_array_R : [],
      dist_time_array_R : [],
      dist_record_R : [],
      dist_time_record_R : [],

      rotate_array_L : [],
      rotate_time_array_L : [],
      rotate_record_L : [],
      rotate_time_record_L : [],
      rotate_array_R : [],
      rotate_time_array_R : [],
      rotate_record_R : [],
      rotate_time_record_R : [],

      fist_array_L : [],
      fist_time_array_L : [],
      fist_record_L : [],
      fist_time_record_L : [],
      fist_array_R : [],
      fist_time_array_R : [],
      fist_record_R : [],
      fist_time_record_R : [],

      still_array_L : [],
      still_time_array_L : [],
      still_record_L : [],
      still_time_record_L : [],
      still_array_R : [],
      still_time_array_R : [],
      still_record_R : [],
      still_time_record_R : [],

      chart_data1_L : null,
      chart_data1_R : null,
      chart_data2_L : null,
      chart_data2_R : null,
      chart_data3_L : null,
      chart_data3_R : null,
      chart_data4_L : null,
      chart_data4_R : null,

      stage:0,
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
      wait:true,
      stage:this.state.stage + 1});
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
        switch(this.state.stage){
          case 1:
            this.setState({hand_dist_array_L:[...this.state.hand_dist_array_L, hand]});
            break;
          case 2:
            this.setState({hand_dist_array_R:[...this.state.hand_dist_array_R, hand]});
            break;
          case 3:
            this.setState({hand_rotate_array_L:[...this.state.hand_rotate_array_L, hand]});
            break;
          case 4:
            this.setState({hand_rotate_array_R:[...this.state.hand_rotate_array_R, hand]});
            break;
          case 5:
            this.setState({hand_fist_array_L:[...this.state.hand_fist_array_L, hand]});
            break;
          case 6:
            this.setState({hand_fist_array_R:[...this.state.hand_fist_array_R, hand]});
            break;
          case 7:
            this.setState({hand_still_array_L:[...this.state.hand_still_array_L, hand]});
            break;
          case 8:
            this.setState({hand_still_array_R:[...this.state.hand_still_array_R, hand]});
            break;
          default:
            break;
        }
        if (hand.length > 0){
          hand.forEach((prediction) => {
            this.setState({dead_frame: 0});
            const landmarks = prediction.landmarks

            let pawn_dist = this.norm(landmarks[0], landmarks[2]);
            let y_dist = this.norm(landmarks[0], landmarks[12]);
            let current_moment = (Date.now() - this.state.startAt)/1000
            let index_dist = null;
            let current_dist = null;
            let rotate_dist = null;
            let fist_dist = null;
            let max_move = null;
            let pinky_rest = null;
            let ring_rest = null;
            let middle_rest = null;
            let index_rest = null;
            let thumb_rest = null;

            switch (this.state.stage){
              case 1:
                drawHand_tap (hand, ctx);
                index_dist = this.norm(landmarks[4], landmarks[8]);
                current_dist = index_dist/pawn_dist;
                this.setState({dist_array_L:[...this.state.dist_array_L, current_dist],
                  dist_time_array_L:[...this.state.dist_time_array_L, current_moment],
                });
                if (this.state.index_passed === 0 && (current_dist - this.state.min_dist) > 0.2){
                  this.setState({index_passed:1,
                    max_dist: current_dist});
                }
                if (this.state.index_passed === 1 && current_dist > this.state.max_dist){
                  this.setState({max_dist: current_dist});
                }
                if (this.state.index_passed === 1 && (this.state.max_dist - current_dist) > 0.2){
                  this.setState({index_passed:0,
                    min_dist: current_dist,
                    tap_count:[...this.state.tap_count, current_moment]});
                }
                if (this.state.index_passed === 0 && current_dist < this.state.min_dist){
                  this.setState({min_dist: current_dist});
                }
                break;
              case 2:
                drawHand_tap (hand, ctx);
                index_dist = this.norm(landmarks[4], landmarks[8]);
                current_dist = index_dist/pawn_dist;
                this.setState({dist_array_R:[...this.state.dist_array_R, current_dist],
                  dist_time_array_R:[...this.state.dist_time_array_R, current_moment],
                });
                if (this.state.index_passed === 0 && (current_dist - this.state.min_dist) > 0.2){
                  this.setState({index_passed:1,
                    max_dist: current_dist});
                }
                if (this.state.index_passed === 1 && current_dist > this.state.max_dist){
                  this.setState({max_dist: current_dist});
                }
                if (this.state.index_passed === 1 && (this.state.max_dist - current_dist) > 0.2){
                  this.setState({index_passed:0,
                    min_dist: current_dist,
                    tap_count:[...this.state.tap_count, current_moment]});
                }
                if (this.state.index_passed === 0 && current_dist < this.state.min_dist){
                  this.setState({min_dist: current_dist});
                }
                break;
              case 3:
                drawHand_rotate (hand, ctx, this.state.rotate_passed);
                rotate_dist = (landmarks[2][0] - landmarks[17][0]) / y_dist;
                this.setState({rotate_array_L:[...this.state.rotate_array_L, rotate_dist],
                  rotate_time_array_L:[...this.state.rotate_time_array_L, current_moment],
                });
                if (this.state.rotate_passed === 0){
                  if (rotate_dist >= 0.2) this.setState({rotate_passed:1});
                  if (rotate_dist <= -0.2) this.setState({rotate_passed:-1});
                }
                if (this.state.rotate_passed === 1 && rotate_dist <= -0.2){
                  this.setState({rotate_passed:-1});
                }
                if (this.state.rotate_passed === -1 && rotate_dist >= 0.2){
                  this.setState({rotate_passed:1,
                    rotate_count:[...this.state.rotate_count, current_moment]});
                }
                break;
              case 4:
                drawHand_rotate (hand, ctx, this.state.rotate_passed);
                rotate_dist = (landmarks[2][0] - landmarks[17][0]) / y_dist;
                this.setState({rotate_array_R:[...this.state.rotate_array_R, rotate_dist],
                  rotate_time_array_R:[...this.state.rotate_time_array_R, current_moment],
                });
                if (this.state.rotate_passed === 0){
                  if (rotate_dist >= 0.2) this.setState({rotate_passed:1});
                  if (rotate_dist <= -0.2) this.setState({rotate_passed:-1});
                }
                if (this.state.rotate_passed === 1 && rotate_dist <= -0.2){
                  this.setState({rotate_passed:-1});
                }
                if (this.state.rotate_passed === -1 && rotate_dist >= 0.2){
                  this.setState({rotate_passed:1,
                    rotate_count:[...this.state.rotate_count, current_moment]});
                }
                break;
              case 5:
                drawHand_fist (hand, ctx, this.state.fist_passed);
                fist_dist =((landmarks[8][1] - landmarks[5][1])+
                            (landmarks[12][1] - landmarks[9][1])+
                            (landmarks[16][1] - landmarks[13][1])+
                            (landmarks[20][1] - landmarks[17][1]))/
                            (4*pawn_dist)
                this.setState({fist_array_L:[...this.state.fist_array_L, fist_dist],
                  fist_time_array_L:[...this.state.fist_time_array_L, current_moment],
                });
                if (fist_dist >= 0.0){this.setState({fist_passed:1})}
                if (fist_dist < -0.4 && this.state.fist_passed === 1){
                  this.setState({fist_passed:0,
                    fist_count:[...this.state.fist_count, current_moment]});
                }
                break;
              case 6:
                drawHand_fist (hand, ctx, this.state.fist_passed);
                fist_dist =((landmarks[8][1] - landmarks[5][1])+
                            (landmarks[12][1] - landmarks[9][1])+
                            (landmarks[16][1] - landmarks[13][1])+
                            (landmarks[20][1] - landmarks[17][1]))/
                            (4*pawn_dist)
                this.setState({fist_array_R:[...this.state.fist_array_R, fist_dist],
                  fist_time_array_R:[...this.state.fist_time_array_R, current_moment],
                });
                if (fist_dist >= 0.0){this.setState({fist_passed:1})}
                if (fist_dist < -0.4 && this.state.fist_passed === 1){
                  this.setState({fist_passed:0,
                    fist_count:[...this.state.fist_count, current_moment]});
                }
                break;
              case 7:
                max_move = 0.0;
                if (this.state.last_hand_L.length > 0){
                  let moved = [];
                  pinky_rest = Math.abs(this.norm(landmarks[0], landmarks[4]) - this.norm(this.state.last_hand_L[0], this.state.last_hand_L[4]))/pawn_dist;
                  ring_rest = Math.abs(this.norm(landmarks[0], landmarks[8]) - this.norm(this.state.last_hand_L[0], this.state.last_hand_L[8]))/pawn_dist;
                  middle_rest = Math.abs(this.norm(landmarks[0], landmarks[12]) - this.norm(this.state.last_hand_L[0], this.state.last_hand_L[12]))/pawn_dist;
                  index_rest = Math.abs(this.norm(landmarks[0], landmarks[16]) - this.norm(this.state.last_hand_L[0], this.state.last_hand_L[16]))/pawn_dist;
                  thumb_rest = Math.abs(this.norm(landmarks[0], landmarks[20]) - this.norm(this.state.last_hand_L[0], this.state.last_hand_L[20]))/pawn_dist;
                  if (pinky_rest > 0.2) moved = [...moved, 4];
                  if (ring_rest > 0.2) moved = [...moved, 8];
                  if (middle_rest > 0.2) moved = [...moved, 12];
                  if (index_rest > 0.2) moved = [...moved, 16];
                  if (thumb_rest > 0.2) moved = [...moved, 20];
                  max_move = Math.max(pinky_rest, ring_rest, middle_rest, index_rest, thumb_rest);
                  if (max_move > this.state.max_tremor_L) this.setState({max_tremor_L:max_move});
                  drawHand_still (hand, ctx, moved);
                }
                else{
                  this.setState({last_hand_L: landmarks});
                }
                this.setState({still_array_L:[...this.state.still_array_L, max_move],
                  still_time_array_L:[...this.state.still_time_array_L, current_moment],
                });
                break;
              case 8:
                max_move = 0.0;
                if (this.state.last_hand_R.length > 0){
                  let moved = [];
                  pinky_rest = Math.abs(this.norm(landmarks[0], landmarks[4]) - this.norm(this.state.last_hand_R[0], this.state.last_hand_R[4]))/pawn_dist;
                  ring_rest = Math.abs(this.norm(landmarks[0], landmarks[8]) - this.norm(this.state.last_hand_R[0], this.state.last_hand_R[8]))/pawn_dist;
                  middle_rest = Math.abs(this.norm(landmarks[0], landmarks[12]) - this.norm(this.state.last_hand_R[0], this.state.last_hand_R[12]))/pawn_dist;
                  index_rest = Math.abs(this.norm(landmarks[0], landmarks[16]) - this.norm(this.state.last_hand_R[0], this.state.last_hand_R[16]))/pawn_dist;
                  thumb_rest = Math.abs(this.norm(landmarks[0], landmarks[20]) - this.norm(this.state.last_hand_R[0], this.state.last_hand_R[20]))/pawn_dist;
                  if (pinky_rest > 0.2) moved = [...moved, 4];
                  if (ring_rest > 0.2) moved = [...moved, 8];
                  if (middle_rest > 0.2) moved = [...moved, 12];
                  if (index_rest > 0.2) moved = [...moved, 16];
                  if (thumb_rest > 0.2) moved = [...moved, 20];
                  max_move = Math.max(pinky_rest, ring_rest, middle_rest, index_rest, thumb_rest);
                  if (max_move > this.state.max_tremor_R) this.setState({max_tremor_R:max_move});
                  drawHand_still (hand, ctx, moved);
                }
                else{
                  this.setState({last_hand_R: landmarks});
                }
                this.setState({still_array_R:[...this.state.still_array_R, max_move],
                  still_time_array_R:[...this.state.still_time_array_R, current_moment],
                });
                break;
              default:
                console.log("Should Not Print");
                break;
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

  async next_step(){
    if (this.state.stage !== 8)
      this.setState({stage:this.state.stage + 1,
        wait:true});
    else{
      clearInterval(this.state.ID);
      if (this.state.recording === true) await this.inference();
      this.setState({real_time_inferencing:false,
        recording:false,
        stage:0});
      this.compose_chart();
      const ctx = this.canvasRef.current.getContext("2d");
      ctx.clearRect(0,0, this.canvasRef.current.width, this.canvasRef.current.height);
    }
  }

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
      wait:true,
      stage:this.state.stage + 1});
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
        //console.log("Waiting till ", this.state.wait_till);
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
          switch (this.state.stage){
            case 1:
              this.setState({dist_record_L:[...this.state.dist_record_L, img],
                dist_time_record_L:[...this.state.dist_time_record_L, current_moment]});
              break;
            case 2:
              this.setState({dist_record_R:[...this.state.dist_record_R, img],
                dist_time_record_R:[...this.state.dist_time_record_R, current_moment]});
              break;
            case 3:
              this.setState({rotate_record_L:[...this.state.rotate_record_L, img],
                rotate_time_record_L:[...this.state.rotate_time_record_L, current_moment]});
              break;
            case 4:
              this.setState({rotate_record_R:[...this.state.rotate_record_R, img],
                rotate_time_record_R:[...this.state.rotate_time_record_R, current_moment]});
              break;
            case 5:
              this.setState({fist_record_L:[...this.state.fist_record_L, img],
                fist_time_record_L:[...this.state.fist_time_record_L, current_moment]});
              break;
            case 6:
              this.setState({fist_record_R:[...this.state.fist_record_R, img],
                fist_time_record_R:[...this.state.fist_time_record_R, current_moment]});
              break;
            case 7:
              this.setState({still_record_L:[...this.state.still_record_L, img],
                still_time_record_L:[...this.state.still_time_record_L, current_moment]});
              break;
            case 8:
              this.setState({still_record_R:[...this.state.still_record_R, img],
                still_time_record_R:[...this.state.still_time_record_R, current_moment]});
              break;
            default:
              console.log('Should print this');
              break;
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
    for (let i = 0; i<this.state.dist_record_L.length; i++){
      const hand = await net.estimateHands(this.state.dist_record_L[i]);
      if (hand.length > 0){
        hand.forEach((prediction) => {
          // Calculate relative distance
          const landmarks = prediction.landmarks;
          let current_moment = (Date.now() - this.state.startAt)/1000
          let index_dist = this.norm(landmarks[4], landmarks[8]);
          let pawn_dist = this.norm(landmarks[0], landmarks[2]);
          let current_dist = index_dist/pawn_dist;
          this.setState({dist_array_L:[...this.state.dist_array_L, current_dist],
            dist_time_array_L:[...this.state.dist_time_array_L, current_moment],
          });
          if (this.state.index_passed === 0 && (current_dist - this.state.min_dist) > 0.2){
            this.setState({index_passed:1,
              max_dist: current_dist});
          }
          if (this.state.index_passed === 1 && current_dist > this.state.max_dist){
            this.setState({max_dist: current_dist});
          }
          if (this.state.index_passed === 1 && (this.state.max_dist - current_dist) > 0.2){
            this.setState({index_passed:0,
              min_dist: current_dist,
              tap_count:[...this.state.tap_count, current_moment]});
          }
          if (this.state.index_passed === 0 && current_dist < this.state.min_dist){
            this.setState({min_dist: current_dist});
          }
          console.log("INDEX COUNT:", this.state.tap_count);
        });
      }
    }
    for (let i = 0; i<this.state.dist_record_R.length; i++){
      const hand = await net.estimateHands(this.state.dist_record_R[i]);
      if (hand.length > 0){
        hand.forEach((prediction) => {
          // Calculate relative distance
          const landmarks = prediction.landmarks;
          let current_moment = (Date.now() - this.state.startAt)/1000
          let index_dist = this.norm(landmarks[4], landmarks[8]);
          let pawn_dist = this.norm(landmarks[0], landmarks[2]);
          let current_dist = index_dist/pawn_dist;
          this.setState({dist_array_R:[...this.state.dist_array_R, current_dist],
            dist_time_array_R:[...this.state.dist_time_array_R, current_moment],
          });
          if (this.state.index_passed === 0 && (current_dist - this.state.min_dist) > 0.2){
            this.setState({index_passed:1,
              max_dist: current_dist});
          }
          if (this.state.index_passed === 1 && current_dist > this.state.max_dist){
            this.setState({max_dist: current_dist});
          }
          if (this.state.index_passed === 1 && (this.state.max_dist - current_dist) > 0.2){
            this.setState({index_passed:0,
              min_dist: current_dist,
              tap_count:[...this.state.tap_count, current_moment]});
          }
          if (this.state.index_passed === 0 && current_dist < this.state.min_dist){
            this.setState({min_dist: current_dist});
          }
          console.log("INDEX COUNT:", this.state.tap_count);
        });
      }
    }
    // Run prediction on recorded rotation data
    for (let i = 0; i<this.state.rotate_record_L.length; i++){
      const hand = await net.estimateHands(this.state.rotate_record_L[i]);
      if (hand.length > 0){
        hand.forEach((prediction) => {
          const landmarks = prediction.landmarks;
          let current_moment = (Date.now() - this.state.startAt)/1000
          let pawn_dist = this.norm(landmarks[0], landmarks[12]);
          let rotate_dist = (landmarks[2][0] - landmarks[17][0]) / pawn_dist;
          this.setState({rotate_array_L:[...this.state.rotate_array_L, rotate_dist],
            rotate_time_array_L:[...this.state.rotate_time_array_L, current_moment],
          });
          if (this.state.rotate_passed === 0){
            if (rotate_dist >= 0.2) this.setState({rotate_passed:1});
            if (rotate_dist <= -0.2) this.setState({rotate_passed:-1});
          }
          if (this.state.rotate_passed === 1 && rotate_dist <= -0.2){
            this.setState({rotate_passed:-1});
          }
          if (this.state.rotate_passed === -1 && rotate_dist >= 0.2){
            this.setState({rotate_passed:1,
              rotate_count:[...this.state.rotate_count, current_moment]});
          }
          console.log("ROTATE COUNT:", this.state.rotate_count);
        });
      }
    }

    for (let i = 0; i<this.state.rotate_record_R.length; i++){
      const hand = await net.estimateHands(this.state.rotate_record_R[i]);
      if (hand.length > 0){
        hand.forEach((prediction) => {
          const landmarks = prediction.landmarks;
          let current_moment = (Date.now() - this.state.startAt)/1000
          let pawn_dist = this.norm(landmarks[0], landmarks[12]);
          let rotate_dist = (landmarks[2][0] - landmarks[17][0]) / pawn_dist;
          this.setState({rotate_array_R:[...this.state.rotate_array_R, rotate_dist],
            rotate_time_array_R:[...this.state.rotate_time_array_R, current_moment],
          });
          if (this.state.rotate_passed === 0){
            if (rotate_dist >= 0.2) this.setState({rotate_passed:1});
            if (rotate_dist <= -0.2) this.setState({rotate_passed:-1});
          }
          if (this.state.rotate_passed === 1 && rotate_dist <= -0.2){
            this.setState({rotate_passed:-1});
          }
          if (this.state.rotate_passed === -1 && rotate_dist >= 0.2){
            this.setState({rotate_passed:1,
              rotate_count:[...this.state.rotate_count, current_moment]});
          }
          console.log("ROTATE COUNT:", this.state.rotate_count);
        });
      }
    }

    // Run prediction on recorded gripping data
    for (let i = 0; i<this.state.fist_record_L.length; i++){
      const hand = await net.estimateHands(this.state.fist_record_L[i]);
      if (hand.length > 0){
        hand.forEach((prediction) => {
          const landmarks = prediction.landmarks;
          let current_moment = (Date.now() - this.state.startAt)/1000
          let pawn_dist = this.norm(landmarks[0], landmarks[2]);
          let fist_dist =((landmarks[8][1] - landmarks[5][1])+
                          (landmarks[12][1] - landmarks[9][1])+
                          (landmarks[16][1] - landmarks[13][1])+
                          (landmarks[20][1] - landmarks[17][1]))/
                          (4*pawn_dist)
          this.setState({fist_array_L:[...this.state.fist_array_L, fist_dist],
            fist_time_array_L:[...this.state.fist_time_array_L, current_moment],
          });
          if (fist_dist >= 0.0){this.setState({fist_passed:1})}
          if (fist_dist < -0.4 && this.state.fist_passed === 1){
            this.setState({fist_passed:0,
              fist_count:[...this.state.fist_count, current_moment]});
          }
          console.log("FIST COUNT:", this.state.fist_count);
        });
      }
    }

    for (let i = 0; i<this.state.fist_record_R.length; i++){
      const hand = await net.estimateHands(this.state.fist_record_R[i]);
      if (hand.length > 0){
        hand.forEach((prediction) => {
          const landmarks = prediction.landmarks;
          let current_moment = (Date.now() - this.state.startAt)/1000;
          let pawn_dist = this.norm(landmarks[0], landmarks[2]);
          let fist_dist =((landmarks[8][1] - landmarks[5][1])+
                          (landmarks[12][1] - landmarks[9][1])+
                          (landmarks[16][1] - landmarks[13][1])+
                          (landmarks[20][1] - landmarks[17][1]))/
                          (4*pawn_dist)
          this.setState({fist_array_R:[...this.state.fist_array_R, fist_dist],
            fist_time_array_R:[...this.state.fist_time_array_R, current_moment],
          });
          if (fist_dist >= 0.0){this.setState({fist_passed:1})}
          if (fist_dist < -0.4 && this.state.fist_passed === 1){
            this.setState({fist_passed:0,
              fist_count:[...this.state.fist_count, current_moment]});
          }
          console.log("FIST COUNT:", this.state.fist_count);
        });
      }
    }

    // Run prediction on recorded postural data
    for (let i = 0; i<this.state.still_record_L.length; i++){
      const hand = await net.estimateHands(this.state.still_record_L[i]);
      if (hand.length > 0){
        hand.forEach((prediction) => {
          const landmarks = prediction.landmarks;
          let current_moment = (Date.now() - this.state.startAt)/1000
          let max_move = 0.0;
          let pawn_dist = this.norm(landmarks[0], landmarks[2]);
          if (this.state.last_hand_L.length > 0){
            let pinky_rest = Math.abs(this.norm(landmarks[0], landmarks[4]) - this.norm(this.state.last_hand_L[0], this.state.last_hand_L[4]))/pawn_dist;
            let ring_rest = Math.abs(this.norm(landmarks[0], landmarks[8]) - this.norm(this.state.last_hand_L[0], this.state.last_hand_L[8]))/pawn_dist;
            let middle_rest = Math.abs(this.norm(landmarks[0], landmarks[12]) - this.norm(this.state.last_hand_L[0], this.state.last_hand_L[12]))/pawn_dist;
            let index_rest = Math.abs(this.norm(landmarks[0], landmarks[16]) - this.norm(this.state.last_hand_L[0], this.state.last_hand_L[16]))/pawn_dist;
            let thumb_rest = Math.abs(this.norm(landmarks[0], landmarks[20]) - this.norm(this.state.last_hand_L[0], this.state.last_hand_L[20]))/pawn_dist;
            max_move = Math.max(pinky_rest, ring_rest, middle_rest, index_rest, thumb_rest);
            if (max_move > this.state.max_tremor_L) this.setState({max_tremor_L:max_move});
          }
          else{
            this.setState({last_hand_L: landmarks});
          }
          this.setState({still_array_L:[...this.state.still_array_L, max_move],
            still_time_array_L:[...this.state.still_time_array_L, current_moment],
          });
        })
      }
    }

    for (let i = 0; i<this.state.still_record_R.length; i++){
      const hand = await net.estimateHands(this.state.still_record_R[i]);
      if (hand.length > 0){
        hand.forEach((prediction) => {
          const landmarks = prediction.landmarks
          let current_moment = (Date.now() - this.state.startAt)/1000
          let max_move = 0.0;
          let pawn_dist = this.norm(landmarks[0], landmarks[2]);
          if (this.state.last_hand_R.length > 0){
            let pinky_rest = Math.abs(this.norm(landmarks[0], landmarks[4]) - this.norm(this.state.last_hand_L[0], this.state.last_hand_L[4]))/pawn_dist;
            let ring_rest = Math.abs(this.norm(landmarks[0], landmarks[8]) - this.norm(this.state.last_hand_L[0], this.state.last_hand_L[8]))/pawn_dist;
            let middle_rest = Math.abs(this.norm(landmarks[0], landmarks[12]) - this.norm(this.state.last_hand_L[0], this.state.last_hand_L[12]))/pawn_dist;
            let index_rest = Math.abs(this.norm(landmarks[0], landmarks[16]) - this.norm(this.state.last_hand_L[0], this.state.last_hand_L[16]))/pawn_dist;
            let thumb_rest = Math.abs(this.norm(landmarks[0], landmarks[20]) - this.norm(this.state.last_hand_L[0], this.state.last_hand_L[20]))/pawn_dist;
            max_move = Math.max(pinky_rest, ring_rest, middle_rest, index_rest, thumb_rest);
            if (max_move > this.state.max_tremor_R) this.setState({max_tremor_R:max_move});
          }
          else{
            this.setState({last_hand_R: landmarks});
          }
          this.setState({still_array_R:[...this.state.still_array_R, max_move],
            still_time_array_R:[...this.state.still_time_array_R, current_moment],
          });
        })
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
      dist_array_L : this.state.dist_array_L,
      dist_time_array_L : this.state.dist_time_array_L,
      rotate_array_L : this.state.rotate_array_L,
      rotate_time_array_L : this.state.rotate_time_array_L,
      fist_array_L : this.state.fist_array_L,
      fist_time_array_L : this.state.fist_time_array_L,
      still_array_L : this.state.still_array_L,
      still_time_array_L : this.state.still_time_array_L,
      hand_dist_array_L : this.state.hand_dist_array_L,
      hand_rotate_array_L : this.state.hand_rotate_array_L,
      hand_fist_array_L : this.state.hand_fist_array_L,
      hand_still_array_L : this.state.hand_still_array_L,
      dist_array_R : this.state.dist_array_R,
      dist_time_array_R : this.state.dist_time_array_R,
      rotate_array_R : this.state.rotate_array_R,
      rotate_time_array_R : this.state.rotate_time_array_R,
      fist_array_R : this.state.fist_array_R,
      fist_time_array_R : this.state.fist_time_array_R,
      still_array_R : this.state.still_array_R,
      still_time_array_R : this.state.still_time_array_R,
      hand_dist_array_R : this.state.hand_dist_array_R,
      hand_rotate_array_R : this.state.hand_rotate_array_R,
      hand_fist_array_R : this.state.hand_fist_array_R,
      hand_still_array_R : this.state.hand_still_array_R,
    }
    let PD = document.getElementById("PD").value;
    let Score = document.getElementById("Score").value;
    console.log(PD, Score);
    this.exportToJson(dict, "state_" + PD + "_" + Score);
  }

  render(){
    const videoConstraints = {
      facingMode: this.state.facingMode
    };
    let button_1 = null;
    let button_2 = null;
    switch(this.state.stage){
      case 1:
        button_1 = <Button disabled={this.state.recording} variant="contained" color="primary"  onClick={this.next_step}>Finish Finger Tapping Left (Real Time)</Button>;
        button_2 = <Button disabled={this.state.real_time_inferencing} variant="contained" color="secondary" onClick={this.next_step}>Finish Finger Tapping Left (Record)</Button>;
        break;
      case 2:
        button_1 = <Button disabled={this.state.recording} variant="contained" color="primary"  onClick={this.next_step}>Finish Finger Tapping Right (Real Time)</Button>;
        button_2 = <Button disabled={this.state.real_time_inferencing} variant="contained" color="secondary" onClick={this.next_step}>Finish Finger Tapping Right (Record)</Button>;
        break;
      case 3:
        button_1 = <Button disabled={this.state.recording} variant="contained" color="primary"  onClick={this.next_step}>Finish Rotation Left (Real Time)</Button>;
        button_2 = <Button disabled={this.state.real_time_inferencing} variant="contained" color="secondary" onClick={this.next_step}>Finish Rotation Left (Record)</Button>;
        break;
      case 4:
        button_1 = <Button disabled={this.state.recording} variant="contained" color="primary"  onClick={this.next_step}>Finish Rotation Right (Real Time)</Button>;
        button_2 = <Button disabled={this.state.real_time_inferencing} variant="contained" color="secondary" onClick={this.next_step}>Finish Rotation Right (Record)</Button>;
        break;
      case 5:
        button_1 = <Button disabled={this.state.recording} variant="contained" color="primary"  onClick={this.next_step}>Finish Gripping Left (Real Time)</Button>;
        button_2 = <Button disabled={this.state.real_time_inferencing} variant="contained" color="secondary" onClick={this.next_step}>Finish Gripping Left (Record)</Button>;
        break;
      case 6:
        button_1 = <Button disabled={this.state.recording} variant="contained" color="primary"  onClick={this.next_step}>Finish Gripping Right (Real Time)</Button>;
        button_2 = <Button disabled={this.state.real_time_inferencing} variant="contained" color="secondary" onClick={this.next_step}>Finish Gripping Right (Record)</Button>;
        break;
      case 7:
        button_1 = <Button disabled={this.state.recording} variant="contained" color="primary"  onClick={this.next_step}>Finish Resting Left (Real Time)</Button>;
        button_2 = <Button disabled={this.state.real_time_inferencing} variant="contained" color="secondary" onClick={this.next_step}>Finish Resting Left (Record)</Button>;
        break;
      case 8:
        button_1 = <Button disabled={this.state.recording} variant="contained" color="primary"  onClick={this.next_step}>Finish Resting Right (Real Time)</Button>;
        button_2 = <Button disabled={this.state.real_time_inferencing} variant="contained" color="secondary" onClick={this.next_step}>Finish Resting Right (Record)</Button>;
        break;
      default:
        button_1 = <Button disabled={this.state.recording} variant="contained" color="primary"  onClick={this.runHandpose}>Start Test (Real Time)</Button>;
        button_2 = <Button disabled={this.state.real_time_inferencing} variant="contained" color="secondary" onClick={this.record_video}>Start Test (Record)</Button>;
        break;
    }
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
            <div>{button_1}</div>
            <div>{button_2}</div>
            <div>
              <Button disabled={this.state.recording||this.state.real_time_inferencing} variant="outlined" color="secondary" onClick={this.reset_counter}>
                Reset All
              </Button>
            </div>
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
            <div>
              <img src={handline} className="handline" alt='anything'/>
            </div>
            <div>
              <small>Enter Vertical Distance (cm)</small>
              <input type="number" id="real_measurement_v" onChange={this.compose_chart} step="0.001" min='0' max='20'></input>
            </div>
            <div>
              <small>Enter Horizontal Distance (cm)</small>
              <input type="number" id="real_measurement_h" onChange={this.compose_chart} step="0.001" min='0' max='20'></input>
            </div>
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
                <Line data={this.state.chart_data1_L} />
                <Line data={this.state.chart_data1_R} />
                <Line data={this.state.chart_data2_L} />
                <Line data={this.state.chart_data2_R} />
                <Line data={this.state.chart_data3_L} />
                <Line data={this.state.chart_data3_R} />
                <Line data={this.state.chart_data4_L} />
                <Line data={this.state.chart_data4_R} />
              </div>
            ) : (
              <div/>
            )}
            <div>
              <small>UPDRS Score, X for N/A, Could be Empty</small>
              <input type="text" id="Score"></input>
            </div>
            <select name="PD" id="PD">
              <option value="False">Control</option>
              <option value="True">PD</option>
            </select>
            <button onClick={this.getArray}>Get Array</button>
            <div>
              <input type="file" id="upload-json"></input>
            </div>
            <div>
              <input type="file" id="upload-weights"></input>
            </div>
          </div>
      </div>
    );
  }
}

export default App;