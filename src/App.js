import React from "react";
import * as handpose from "@tensorflow-models/handpose";
import "./App.css";
import { sqrt, pow } from "mathjs"

class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      ID:null,
      hand_array:[],
      time_array:[],
      result_array:[],
      startAt: Date.now(),
    };
    this.canvasRef = React.createRef(null);
    this.sleep = this.sleep.bind(this);
    this.inference = this.inference.bind(this);
    this.norm = this.norm.bind(this);
    this.getArray = this.getArray.bind(this);
    this.exportToJson = this.exportToJson.bind(this);
    this.record_video = this.record_video.bind(this);
    this.stop_record = this.stop_record.bind(this);
    this.reset_record = this.reset_record.bind(this);
    require('@tensorflow/tfjs-backend-webgl');
    this.net = null;
  }

  loadVideo = (event) =>{
    var url = URL.createObjectURL(event.target.files[0]);
    this.setState({videoPath:url});
    var video = document.getElementById('user-video');
    video.src = url;
    video.onloadeddata = () =>{
      video.play();
    }
    video.load();
  };

  async record_video(){
    console.log("Start Recording");
    this.setState({startAt:Date.now()});
    const Interval_ID = setInterval(() => {
      this.concat_frame();
    }, 50);
    this.setState({ID:Interval_ID,});
  }

  async concat_frame() {
    const video = document.getElementById("user-video");
    this.canvasRef.current.width = video.videoWidth;
    this.canvasRef.current.height = video.videoHeight;
    const ctx = this.canvasRef.current.getContext("2d");
    ctx.drawImage(video, 0, 0, this.canvasRef.current.width, this.canvasRef.current.height);
    let current_moment = (Date.now() - this.state.startAt)/1000;
    var img = document.createElement("img");
    img.src = this.canvasRef.current.toDataURL();
    img.onload = function(){
      this.setState({hand_array:[...this.state.hand_array, img],
                     time_array:[...this.state.time_array, current_moment]});
    }.bind(this)
    console.log(this.state.hand_array);
  }

  async stop_record() {
    console.log("Stop Recording");
    clearInterval(this.state.ID);
    const video = document.getElementById("user-video");
    video.pause();
    await this.inference();
    console.log("Inference Finished");
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
    const directory = document.getElementById("fname").value;
    let dict = {
      time: this.state.time_array,
      result: this.state.result_array,
    }
    console.log(dict);
    this.exportToJson(dict, directory);
  }

  norm = (lm1, lm2) => {
    return sqrt(pow(lm1[0]-lm2[0], 2)+pow(lm1[1]-lm2[1], 2))
  }  

  reset_record = async() =>{
    console.log("Reseting Arrays");
    await this.setState({
      hand_array:[],
      time_array:[],
      result_array:[],
    });
    this.sleep(100);
    console.log("Hands: ", this.state.hand_array);
    console.log("Result: ", this.state.result_array);
  }

  sleep = (milliseconds) => {
    return new Promise(resolve => setTimeout(resolve, milliseconds))
  }  

  inference = async() =>{
    this.sleep(300);
    require('@tensorflow/tfjs-backend-webgl');
    const net = await handpose.load();
    console.log("Handpose model loaded.");

    // Run prediction on recorded tapping data
    for (let i = 0; i<this.state.hand_array.length; i++){
      const hand = await net.estimateHands(this.state.hand_array[i]);
      if (hand.length > 0){
        hand.forEach((prediction) => {
          this.setState({result_array:[...this.state.result_array, prediction]});
        });
      }
      else{
        this.setState({result_array:[...this.state.result_array, null]});
      }
    }
  }

  /*
  checkImg = (str) =>{
    if (str.charAt(str.length - 1) === 'g' && 
    str.charAt(str.length - 2) === 'p' &&
    str.charAt(str.length - 3) === 'j' ){
      return true;
    }
  }

  useRecord = async() =>{
    require('@tensorflow/tfjs-backend-webgl');
    const net = await handpose.load();
    for (let j in this.state.urls) {
      if (this.checkImg(this.state.urls[j])) {
        await this.inference(this.state.urls[j], net);
        await this.sleep(100);
      }
    }
    console.log("FINISH");
    return true;
  }

  real_time_handpose = async() =>{
    const Interval_ID = setInterval(() => {
      this.paint();
    }, 50);
    this.setState({ID:Interval_ID});
  } 

  runHandpose = async () => {
    this.setState({urls:[]});
    const $ = require('cheerio');
    const rp = require('request-promise');
    const directory = document.getElementById("fname").value
    const url = 'http://localhost:8000/' + directory;
    console.log("Looking for image in: ", url);
    const links = [];
    rp(url).then(html => {
        const linkObjects = $('a', html);
        const total = linkObjects.length;
        for(let i = 0; i < total; i++){
            links.push({
                href: directory+"/"+linkObjects[i].attribs.href,
                title: linkObjects[i].attribs.title
            });
        }
        for (let j in links) this.setState({urls:[...this.state.urls, links[j].href]});
        this.state.urls.sort(function(x, y) {
          x = x.split('/')[1].split('.')[0];
          y = y.split('/')[1].split('.')[0];
          let int_x = parseInt(x);
          let int_y = parseInt(y);
          if (int_x < int_y) {
            return -1;
          }
          if (int_x > int_y) {
            return 1;
          }
          return 0;
        });
        console.log(this.state.urls);
    })
    .catch(err => {
        console.log(err); 
    })
  };

  load = async() =>{
    this.net = await handpose.load();
    console.log("Loaded Handpose");
  }

  paint = async() => {
    const video = document.getElementById("user-video");
    this.canvasRef.current.width = video.videoWidth;
    this.canvasRef.current.height = video.videoHeight;
    const ctx = this.canvasRef.current.getContext("2d");
    ctx.drawImage(video, 0, 0, this.canvasRef.current.width, this.canvasRef.current.height);
    //const dataUrl = this.canvasRef.current.toDataURL();
    const hand = await this.net.estimateHands(video);
    if (hand.length > 0){
      hand.forEach((prediction) => {
        drawHand(hand, ctx);
        this.setState({hand_array:[...this.state.hand_array, hand]});
        const landmarks = prediction.landmarks
        let pawn_dist = this.norm(landmarks[0], landmarks[2]);
        let current_moment = (Date.now() - this.state.startAt)/1000
        let index_dist = this.norm(landmarks[4], landmarks[8]);
        let current_dist = index_dist/pawn_dist;
        console.log(current_moment, current_dist);
        if (this.state.index_passed === 0 && (current_dist - this.state.min_dist) > 0.5){
          console.log("Up Pass", current_moment);
          this.setState({index_passed:1,
            max_dist: current_dist});
        }
        if (this.state.index_passed === 1 && current_dist > this.state.max_dist){
          this.setState({max_dist: current_dist});
        }
        if (this.state.index_passed === 1 && (this.state.max_dist - current_dist) > 0.5){
          console.log("Down Pass", current_moment);
          this.setState({index_passed:0,
            min_dist: current_dist,
            tap_count:[...this.state.tap_count, current_moment]});
        }
        if (this.state.index_passed === 0 && current_dist < this.state.min_dist){
          this.setState({min_dist: current_dist});
        }
      });
    }
  }
  */

  render(){
    return (
      <div className="App">
        <header className="App-header"></header>
        <div>
          <div>
            <video id="user-video" crossOrigin="anonymous" controls="controls">
              <source id='source' type="video/mp4"></source>
            </video>
          </div>
          <div>
            <input type="file" id="videof" onChange={this.loadVideo}/>
          </div>
          <button onClick={this.record_video}>1.Record</button>
          <button onClick={this.stop_record}>2.Run Inference</button>
          <button onClick={this.reset_record}>4.Reset Recording</button>
          <div>
            <input type="text" id="fname"/>
            <button onClick={this.getArray}>3.Get Array</button>   
          </div>
        </div>
        <div>
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
        </div>
      </div>
    );
  }
}

export default App;