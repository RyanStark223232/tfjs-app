(this.webpackJsonpfinger=this.webpackJsonpfinger||[]).push([[0],{298:function(t,e,a){},304:function(t,e){},305:function(t,e){},313:function(t,e){},316:function(t,e,a){},415:function(t,e,a){"use strict";a.r(e);var r=a(36),i=a(18),s=a.n(i),n=a(139),o=a.n(n),c=(a(298),a(14)),d=a(11),h=a.n(d),_=a(15),l=a(10),u=a(16),f=a(40),p=a(45),b=a(44),m=a(433),g=a(217),v=a(205),y=a(289),j=a(290),w=a.n(j),O=(a(316),a(6)),S="aqua",x={0:{color:"gold",size:15},1:{color:"gold",size:6},2:{color:"gold",size:10},3:{color:"gold",size:6},4:{color:"red",size:6},5:{color:"gold",size:10},6:{color:"gold",size:6},7:{color:"gold",size:6},8:{color:"red",size:6},9:{color:"gold",size:10},10:{color:"gold",size:6},11:{color:"gold",size:6},12:{color:"gold",size:6},13:{color:"gold",size:10},14:{color:"gold",size:6},15:{color:"gold",size:6},16:{color:"gold",size:6},17:{color:"gold",size:10},18:{color:"gold",size:6},19:{color:"gold",size:6},20:{color:"gold",size:6}},k=function(t,e){var a=arguments.length>2&&void 0!==arguments[2]?arguments[2]:{},r=e.text,i=e.x,s=e.y,n=a.fontSize,o=void 0===n?50:n,c=a.fontFamily,d=void 0===c?"Arial":c,h=a.color,_=void 0===h?"red":h,l=a.textAlign,u=void 0===l?"center":l,f=a.textBaseline,p=void 0===f?"center":f;t.beginPath(),t.font=o+"px "+d,t.textAlign=u,t.textBaseline=p,t.fillStyle=_,t.fillText(r,i,s),t.stroke()},R=function(t,e){t.length>0&&t.forEach((function(t){var a=t.landmarks;e.beginPath(),e.moveTo(a[8][0],a[8][1]),e.lineTo(a[4][0],a[4][1]),e.strokeStyle="plum",e.lineWidth=4,e.stroke();for(var r=0;r<a.length;r++){var i=a[r][0],s=a[r][1];e.beginPath(),e.arc(i,s,x[r].size,0,3*Math.PI),e.fillStyle=x[r].color,e.fill()}}))};function C(t){return[t.y,t.x]}function D(t,e,a,r,i){t.beginPath(),t.arc(a,e,r,0,2*Math.PI),t.fillStyle=i,t.fill()}function H(t,e,a,r,i){var s=Object(O.a)(t,2),n=s[0],o=s[1],c=Object(O.a)(e,2),d=c[0],h=c[1];i.beginPath(),i.moveTo(o*r,n*r),i.lineTo(h*r,d*r),i.lineWidth=2,i.strokeStyle=a,i.stroke()}function I(t,e,a){var r=arguments.length>3&&void 0!==arguments[3]?arguments[3]:1,i=v.a(t,e);i.forEach((function(t){H(C(t[0].position),C(t[1].position),S,r,a)}))}function z(t,e,a){for(var r=arguments.length>3&&void 0!==arguments[3]?arguments[3]:1,i=0;i<t.length;i++){var s=t[i];if(!(s.score<e)){var n=s.position,o=n.y,c=n.x;D(a,o*r,c*r,3,S)}}}var B=[127,34,139,11,0,37,232,231,120,72,37,39,128,121,47,232,121,128,104,69,67,175,171,148,157,154,155,118,50,101,73,39,40,9,151,108,48,115,131,194,204,211,74,40,185,80,42,183,40,92,186,230,229,118,202,212,214,83,18,17,76,61,146,160,29,30,56,157,173,106,204,194,135,214,192,203,165,98,21,71,68,51,45,4,144,24,23,77,146,91,205,50,187,201,200,18,91,106,182,90,91,181,85,84,17,206,203,36,148,171,140,92,40,39,193,189,244,159,158,28,247,246,161,236,3,196,54,68,104,193,168,8,117,228,31,189,193,55,98,97,99,126,47,100,166,79,218,155,154,26,209,49,131,135,136,150,47,126,217,223,52,53,45,51,134,211,170,140,67,69,108,43,106,91,230,119,120,226,130,247,63,53,52,238,20,242,46,70,156,78,62,96,46,53,63,143,34,227,173,155,133,123,117,111,44,125,19,236,134,51,216,206,205,154,153,22,39,37,167,200,201,208,36,142,100,57,212,202,20,60,99,28,158,157,35,226,113,160,159,27,204,202,210,113,225,46,43,202,204,62,76,77,137,123,116,41,38,72,203,129,142,64,98,240,49,102,64,41,73,74,212,216,207,42,74,184,169,170,211,170,149,176,105,66,69,122,6,168,123,147,187,96,77,90,65,55,107,89,90,180,101,100,120,63,105,104,93,137,227,15,86,85,129,102,49,14,87,86,55,8,9,100,47,121,145,23,22,88,89,179,6,122,196,88,95,96,138,172,136,215,58,172,115,48,219,42,80,81,195,3,51,43,146,61,171,175,199,81,82,38,53,46,225,144,163,110,246,33,7,52,65,66,229,228,117,34,127,234,107,108,69,109,108,151,48,64,235,62,78,191,129,209,126,111,35,143,163,161,246,117,123,50,222,65,52,19,125,141,221,55,65,3,195,197,25,7,33,220,237,44,70,71,139,122,193,245,247,130,33,71,21,162,153,158,159,170,169,150,188,174,196,216,186,92,144,160,161,2,97,167,141,125,241,164,167,37,72,38,12,145,159,160,38,82,13,63,68,71,226,35,111,158,153,154,101,50,205,206,92,165,209,198,217,165,167,97,220,115,218,133,112,243,239,238,241,214,135,169,190,173,133,171,208,32,125,44,237,86,87,178,85,86,179,84,85,180,83,84,181,201,83,182,137,93,132,76,62,183,61,76,184,57,61,185,212,57,186,214,207,187,34,143,156,79,239,237,123,137,177,44,1,4,201,194,32,64,102,129,213,215,138,59,166,219,242,99,97,2,94,141,75,59,235,24,110,228,25,130,226,23,24,229,22,23,230,26,22,231,112,26,232,189,190,243,221,56,190,28,56,221,27,28,222,29,27,223,30,29,224,247,30,225,238,79,20,166,59,75,60,75,240,147,177,215,20,79,166,187,147,213,112,233,244,233,128,245,128,114,188,114,217,174,131,115,220,217,198,236,198,131,134,177,132,58,143,35,124,110,163,7,228,110,25,356,389,368,11,302,267,452,350,349,302,303,269,357,343,277,452,453,357,333,332,297,175,152,377,384,398,382,347,348,330,303,304,270,9,336,337,278,279,360,418,262,431,304,408,409,310,415,407,270,409,410,450,348,347,422,430,434,313,314,17,306,307,375,387,388,260,286,414,398,335,406,418,364,367,416,423,358,327,251,284,298,281,5,4,373,374,253,307,320,321,425,427,411,421,313,18,321,405,406,320,404,405,315,16,17,426,425,266,377,400,369,322,391,269,417,465,464,386,257,258,466,260,388,456,399,419,284,332,333,417,285,8,346,340,261,413,441,285,327,460,328,355,371,329,392,439,438,382,341,256,429,420,360,364,394,379,277,343,437,443,444,283,275,440,363,431,262,369,297,338,337,273,375,321,450,451,349,446,342,467,293,334,282,458,461,462,276,353,383,308,324,325,276,300,293,372,345,447,382,398,362,352,345,340,274,1,19,456,248,281,436,427,425,381,256,252,269,391,393,200,199,428,266,330,329,287,273,422,250,462,328,258,286,384,265,353,342,387,259,257,424,431,430,342,353,276,273,335,424,292,325,307,366,447,345,271,303,302,423,266,371,294,455,460,279,278,294,271,272,304,432,434,427,272,407,408,394,430,431,395,369,400,334,333,299,351,417,168,352,280,411,325,319,320,295,296,336,319,403,404,330,348,349,293,298,333,323,454,447,15,16,315,358,429,279,14,15,316,285,336,9,329,349,350,374,380,252,318,402,403,6,197,419,318,319,325,367,364,365,435,367,397,344,438,439,272,271,311,195,5,281,273,287,291,396,428,199,311,271,268,283,444,445,373,254,339,263,466,249,282,334,296,449,347,346,264,447,454,336,296,299,338,10,151,278,439,455,292,407,415,358,371,355,340,345,372,390,249,466,346,347,280,442,443,282,19,94,370,441,442,295,248,419,197,263,255,359,440,275,274,300,383,368,351,412,465,263,467,466,301,368,389,380,374,386,395,378,379,412,351,419,436,426,322,373,390,388,2,164,393,370,462,461,164,0,267,302,11,12,374,373,387,268,12,13,293,300,301,446,261,340,385,384,381,330,266,425,426,423,391,429,355,437,391,327,326,440,457,438,341,382,362,459,457,461,434,430,394,414,463,362,396,369,262,354,461,457,316,403,402,315,404,403,314,405,404,313,406,405,421,418,406,366,401,361,306,408,407,291,409,408,287,410,409,432,436,410,434,416,411,264,368,383,309,438,457,352,376,401,274,275,4,421,428,262,294,327,358,433,416,367,289,455,439,462,370,326,2,326,370,305,460,455,254,449,448,255,261,446,253,450,449,252,451,450,256,452,451,341,453,452,413,464,463,441,413,414,258,442,441,257,443,442,259,444,443,260,445,444,467,342,445,459,458,250,289,392,290,290,328,460,376,433,435,250,290,392,411,416,433,341,463,464,453,464,465,357,465,412,343,412,399,360,363,440,437,399,456,420,456,363,401,435,288,372,383,353,339,255,249,448,261,255,133,243,190,133,155,112,33,246,247,33,130,25,398,384,286,362,398,414,362,463,341,263,359,467,263,249,255,466,467,260,75,60,166,238,239,79,162,127,139,72,11,37,121,232,120,73,72,39,114,128,47,233,232,128,103,104,67,152,175,148,173,157,155,119,118,101,74,73,40,107,9,108,49,48,131,32,194,211,184,74,185,191,80,183,185,40,186,119,230,118,210,202,214,84,83,17,77,76,146,161,160,30,190,56,173,182,106,194,138,135,192,129,203,98,54,21,68,5,51,4,145,144,23,90,77,91,207,205,187,83,201,18,181,91,182,180,90,181,16,85,17,205,206,36,176,148,140,165,92,39,245,193,244,27,159,28,30,247,161,174,236,196,103,54,104,55,193,8,111,117,31,221,189,55,240,98,99,142,126,100,219,166,218,112,155,26,198,209,131,169,135,150,114,47,217,224,223,53,220,45,134,32,211,140,109,67,108,146,43,91,231,230,120,113,226,247,105,63,52,241,238,242,124,46,156,95,78,96,70,46,63,116,143,227,116,123,111,1,44,19,3,236,51,207,216,205,26,154,22,165,39,167,199,200,208,101,36,100,43,57,202,242,20,99,56,28,157,124,35,113,29,160,27,211,204,210,124,113,46,106,43,204,96,62,77,227,137,116,73,41,72,36,203,142,235,64,240,48,49,64,42,41,74,214,212,207,183,42,184,210,169,211,140,170,176,104,105,69,193,122,168,50,123,187,89,96,90,66,65,107,179,89,180,119,101,120,68,63,104,234,93,227,16,15,85,209,129,49,15,14,86,107,55,9,120,100,121,153,145,22,178,88,179,197,6,196,89,88,96,135,138,136,138,215,172,218,115,219,41,42,81,5,195,51,57,43,61,208,171,199,41,81,38,224,53,225,24,144,110,105,52,66,118,229,117,227,34,234,66,107,69,10,109,151,219,48,235,183,62,191,142,129,126,116,111,143,7,163,246,118,117,50,223,222,52,94,19,141,222,221,65,196,3,197,45,220,44,156,70,139,188,122,245,139,71,162,145,153,159,149,170,150,122,188,196,206,216,92,163,144,161,164,2,167,242,141,241,0,164,37,11,72,12,144,145,160,12,38,13,70,63,71,31,226,111,157,158,154,36,101,205,203,206,165,126,209,217,98,165,97,237,220,218,237,239,241,210,214,169,140,171,32,241,125,237,179,86,178,180,85,179,181,84,180,182,83,181,194,201,182,177,137,132,184,76,183,185,61,184,186,57,185,216,212,186,192,214,187,139,34,156,218,79,237,147,123,177,45,44,4,208,201,32,98,64,129,192,213,138,235,59,219,141,242,97,97,2,141,240,75,235,229,24,228,31,25,226,230,23,229,231,22,230,232,26,231,233,112,232,244,189,243,189,221,190,222,28,221,223,27,222,224,29,223,225,30,224,113,247,225,99,60,240,213,147,215,60,20,166,192,187,213,243,112,244,244,233,245,245,128,188,188,114,174,134,131,220,174,217,236,236,198,134,215,177,58,156,143,124,25,110,7,31,228,25,264,356,368,0,11,267,451,452,349,267,302,269,350,357,277,350,452,357,299,333,297,396,175,377,381,384,382,280,347,330,269,303,270,151,9,337,344,278,360,424,418,431,270,304,409,272,310,407,322,270,410,449,450,347,432,422,434,18,313,17,291,306,375,259,387,260,424,335,418,434,364,416,391,423,327,301,251,298,275,281,4,254,373,253,375,307,321,280,425,411,200,421,18,335,321,406,321,320,405,314,315,17,423,426,266,396,377,369,270,322,269,413,417,464,385,386,258,248,456,419,298,284,333,168,417,8,448,346,261,417,413,285,326,327,328,277,355,329,309,392,438,381,382,256,279,429,360,365,364,379,355,277,437,282,443,283,281,275,363,395,431,369,299,297,337,335,273,321,348,450,349,359,446,467,283,293,282,250,458,462,300,276,383,292,308,325,283,276,293,264,372,447,346,352,340,354,274,19,363,456,281,426,436,425,380,381,252,267,269,393,421,200,428,371,266,329,432,287,422,290,250,328,385,258,384,446,265,342,386,387,257,422,424,430,445,342,276,422,273,424,306,292,307,352,366,345,268,271,302,358,423,371,327,294,460,331,279,294,303,271,304,436,432,427,304,272,408,395,394,431,378,395,400,296,334,299,6,351,168,376,352,411,307,325,320,285,295,336,320,319,404,329,330,349,334,293,333,366,323,447,316,15,315,331,358,279,317,14,316,8,285,9,277,329,350,253,374,252,319,318,403,351,6,419,324,318,325,397,367,365,288,435,397,278,344,439,310,272,311,248,195,281,375,273,291,175,396,199,312,311,268,276,283,445,390,373,339,295,282,296,448,449,346,356,264,454,337,336,299,337,338,151,294,278,455,308,292,415,429,358,355,265,340,372,388,390,466,352,346,280,295,442,282,354,19,370,285,441,295,195,248,197,457,440,274,301,300,368,417,351,465,251,301,389,385,380,386,394,395,379,399,412,419,410,436,322,387,373,388,326,2,393,354,370,461,393,164,267,268,302,12,386,374,387,312,268,13,298,293,301,265,446,340,380,385,381,280,330,425,322,426,391,420,429,437,393,391,326,344,440,438,458,459,461,364,434,394,428,396,262,274,354,457,317,316,402,316,315,403,315,314,404,314,313,405,313,421,406,323,366,361,292,306,407,306,291,408,291,287,409,287,432,410,427,434,411,372,264,383,459,309,457,366,352,401,1,274,4,418,421,262,331,294,358,435,433,367,392,289,439,328,462,326,94,2,370,289,305,455,339,254,448,359,255,446,254,253,449,253,252,450,252,256,451,256,341,452,414,413,463,286,441,414,286,258,441,258,257,442,257,259,443,259,260,444,260,467,445,309,459,250,305,289,290,305,290,460,401,376,435,309,250,392,376,411,433,453,341,464,357,453,465,343,357,412,437,343,399,344,360,440,420,437,456,360,420,363,361,401,288,265,372,353,390,339,249,339,448,255],T=function(t,e,a){var r=new Path2D;r.moveTo(e[0][0],e[0][1]);for(var i=1;i<e.length;i++){var s=e[i];r.lineTo(s[0],s[1])}a&&r.closePath(),t.strokeStyle="grey",t.stroke(r)},F=function(t,e){t.length>0&&t.forEach((function(t){for(var a=t.scaledMesh,r=0;r<B.length/3;r++){var i=[B[3*r],B[3*r+1],B[3*r+2]].map((function(t){return a[t]}));T(e,i,!0)}for(var s=0;s<a.length;s++){var n=a[s][0],o=a[s][1];e.beginPath(),e.arc(n,o,1,0,3*Math.PI),e.fillStyle="aqua",e.fill()}}))},P=a(432),A=a(207),M=function(t){Object(p.a)(i,t);var e=Object(b.a)(i);function i(t){var a;return Object(l.a)(this,i),(a=e.call(this,t)).sleep=function(t){return new Promise((function(e){return setTimeout(e,t)}))},a.state={ID:0,tap_count:[],rotate_count:[],fist_count:[],index_passed:0,rotate_passed:0,last_pressed:0,real_time_inferencing:!1,recording:!1,button_mode:!1,chart_ready:!1,finger_done:!1,rotate_done:!1,fist_done:!1,pawn_dist_array:[],pawn_rotate_array:[],pawn_fist_array:[],dist_array:[],dist_time_array:[],dist_record:[],dist_time_record:[],rotate_array:[],rotate_time_array:[],rotate_record:[],rotate_time_record:[],fist_array:[],fist_time_array:[],fist_record:[],fist_time_record:[],gait_record:[],gait_time_record:[],chart_data1:null,chart_data2:null,chart_data3:null,wait:!1,wait_till:0,startAt:Date.now(),dead_frame:0,ctx:null,raw:!0,facingMode:"user"},a.webcamRef=s.a.createRef(null),a.canvasRef=s.a.createRef(null),a.videoConstraints={facingMode:"user"},a.runHandpose=a.runHandpose.bind(Object(f.a)(a)),a.stop_real_time_inference=a.stop_real_time_inference.bind(Object(f.a)(a)),a.stop_tapping=a.stop_tapping.bind(Object(f.a)(a)),a.stop_rotating=a.stop_rotating.bind(Object(f.a)(a)),a.stop_record=a.stop_record.bind(Object(f.a)(a)),a.record_video=a.record_video.bind(Object(f.a)(a)),a.concat_frame=a.concat_frame.bind(Object(f.a)(a)),a.inference=a.inference.bind(Object(f.a)(a)),a.reset_counter=a.reset_counter.bind(Object(f.a)(a)),a.increment_tap1=a.increment_tap1.bind(Object(f.a)(a)),a.increment_tap2=a.increment_tap2.bind(Object(f.a)(a)),a.switch_button=a.switch_button.bind(Object(f.a)(a)),a.compose_chart=a.compose_chart.bind(Object(f.a)(a)),a.runPosenet=a.runPosenet.bind(Object(f.a)(a)),a.read_time_posenet=a.read_time_posenet.bind(Object(f.a)(a)),a.runFacemesh=a.runFacemesh.bind(Object(f.a)(a)),a.read_time_facemesh=a.read_time_facemesh.bind(Object(f.a)(a)),a.switch_style=a.switch_style.bind(Object(f.a)(a)),a.switch_cam=a.switch_cam.bind(Object(f.a)(a)),a}return Object(u.a)(i,[{key:"switch_cam",value:function(){"user"===this.state.facingMode?this.setState({facingMode:"environment"}):"environment"===this.state.facingMode&&this.setState({facingMode:"user"})}},{key:"runFacemesh",value:function(){var t=Object(_.a)(h.a.mark((function t(){var e,r,i=this;return h.a.wrap((function(t){for(;;)switch(t.prev=t.next){case 0:return a(162),t.next=3,y.a();case 3:e=t.sent,console.log("Facemesh model loaded."),this.setState({startAt:Date.now()}),r=setInterval((function(){i.read_time_facemesh(e)}),50),this.setState({ID:r}),this.setState({real_time_inferencing:!0});case 9:case"end":return t.stop()}}),t,this)})));return function(){return t.apply(this,arguments)}}()},{key:"read_time_facemesh",value:function(){var t=Object(_.a)(h.a.mark((function t(e){var a,r,i,s,n;return h.a.wrap((function(t){for(;;)switch(t.prev=t.next){case 0:if(this.state.wait&&(this.setState({wait_till:Date.now()+3e3}),this.setState({wait:!1})),"undefined"===typeof this.webcamRef.current||null===this.webcamRef.current||4!==this.webcamRef.current.video.readyState){t.next=14;break}return a=this.webcamRef.current.video,r=this.webcamRef.current.video.videoWidth,i=this.webcamRef.current.video.videoHeight,this.webcamRef.current.video.width=r,this.webcamRef.current.video.height=i,this.canvasRef.current.width=r,this.canvasRef.current.height=i,t.next=11,e.estimateFaces(a);case 11:s=t.sent,n=this.canvasRef.current.getContext("2d"),F(s,n);case 14:case"end":return t.stop()}}),t,this)})));return function(e){return t.apply(this,arguments)}}()},{key:"runPosenet",value:function(){var t=Object(_.a)(h.a.mark((function t(){var e,r,i=this;return h.a.wrap((function(t){for(;;)switch(t.prev=t.next){case 0:return a(162),t.next=3,v.b();case 3:e=t.sent,console.log("PoseNet model loaded."),this.setState({startAt:Date.now()}),r=setInterval((function(){i.read_time_posenet(e)}),50),this.setState({ID:r}),this.setState({real_time_inferencing:!0});case 9:case"end":return t.stop()}}),t,this)})));return function(){return t.apply(this,arguments)}}()},{key:"read_time_posenet",value:function(){var t=Object(_.a)(h.a.mark((function t(e){var a,r,i,s,n;return h.a.wrap((function(t){for(;;)switch(t.prev=t.next){case 0:if(this.state.wait&&(this.setState({wait_till:Date.now()+3e3}),this.setState({wait:!1})),"undefined"===typeof this.webcamRef.current||null===this.webcamRef.current||4!==this.webcamRef.current.video.readyState){t.next=15;break}return a=this.webcamRef.current.video,r=this.webcamRef.current.video.videoWidth,i=this.webcamRef.current.video.videoHeight,this.webcamRef.current.video.width=r,this.webcamRef.current.video.height=i,this.canvasRef.current.width=r,this.canvasRef.current.height=i,t.next=11,e.estimateSinglePose(a);case 11:s=t.sent,n=this.canvasRef.current.getContext("2d"),z(s.keypoints,.6,n),I(s.keypoints,.7,n);case 15:case"end":return t.stop()}}),t,this)})));return function(e){return t.apply(this,arguments)}}()},{key:"switch_style",value:function(){var t=Object(_.a)(h.a.mark((function t(){return h.a.wrap((function(t){for(;;)switch(t.prev=t.next){case 0:return t.next=2,this.setState({raw:!this.state.raw});case 2:this.compose_chart();case 3:case"end":return t.stop()}}),t,this)})));return function(){return t.apply(this,arguments)}}()},{key:"compose_chart",value:function(){var t=[],e=[],a=[],r=[],i=[],s=[],n=document.getElementById("real_measurement").value;if(this.state.raw){if(t=Object(c.a)(this.state.dist_time_array),e=Object(c.a)(this.state.dist_array),a=Object(c.a)(this.state.rotate_time_array),r=Object(c.a)(this.state.rotate_array),i=Object(c.a)(this.state.fist_time_array),s=Object(c.a)(this.state.fist_array),n>1e-4){var o=0,d=0;for(o=0;o<e.length;o++)d=e[o],e[o]=d*n;for(o=0;o<r.length;o++)d=r[o],r[o]=d*n;for(o=0;o<s.length;o++)d=s[o],s[o]=d*n}}else{for(var h=this.state.dist_time_array[0],_=this.state.dist_time_array[this.state.dist_time_array.length-1];h+1<_;){t=[].concat(Object(c.a)(t),[h]);for(var l=0,u=0,f=0;f<this.state.dist_array.length;f++)this.state.dist_time_array[f]>=h&&this.state.dist_time_array[f]<h+1&&(u<this.state.dist_array[f]&&(u=this.state.dist_array[f]),this.state.tap_count.includes(this.state.dist_time_array[f])&&(l+=u,u=0));e=[].concat(Object(c.a)(e),[l]),h+=.1}for(h=this.state.rotate_time_array[0],_=this.state.rotate_time_array[this.state.rotate_time_array.length-1];h+5<_;){a=[].concat(Object(c.a)(a),[h+2.5]);var p=0,b=0;for(var m in this.state.rotate_count)(b=this.state.rotate_count[m])>=h&&b<h+5&&(p+=1);r=[].concat(Object(c.a)(r),[p]),h+=.1}for(h=this.state.fist_time_array[0],_=this.state.fist_time_array[this.state.fist_time_array.length-1];h+5<_;){i=[].concat(Object(c.a)(i),[h+2.5]);var g=0,v=0;for(var y in this.state.fist_count)(v=this.state.fist_count[y])>=h&&v<h+5&&(g+=1);s=[].concat(Object(c.a)(s),[g]),h+=.1}}var j={labels:t,datasets:[{label:"Tapping",fill:!1,lineTension:.3,backgroundColor:"rgba(75,192,192,0.4)",borderColor:"rgba(75,192,192,1)",borderCapStyle:"butt",borderDash:[],borderDashOffset:0,borderJoinStyle:"miter",pointBorderColor:"rgba(75,192,192,1)",pointBackgroundColor:"#fff",pointBorderWidth:1,pointHoverRadius:5,pointHoverBackgroundColor:"rgba(75,192,192,1)",pointHoverBorderColor:"rgba(220,220,220,1)",pointHoverBorderWidth:2,pointRadius:1,pointHitRadius:10,data:e}]},w={labels:a,datasets:[{label:"Rotation",fill:!1,lineTension:.1,backgroundColor:"rgba(192,75,192,0.4)",borderColor:"rgba(192,75,192,1)",borderCapStyle:"butt",borderDash:[],borderDashOffset:0,borderJoinStyle:"miter",pointBorderColor:"rgba(192,75,192,1)",pointBackgroundColor:"#fff",pointBorderWidth:1,pointHoverRadius:5,pointHoverBackgroundColor:"rgba(192,75,192,1)",pointHoverBorderColor:"rgba(220,220,220,1)",pointHoverBorderWidth:2,pointRadius:1,pointHitRadius:10,data:r}]},O={labels:i,datasets:[{label:"Fist",fill:!1,lineTension:.1,backgroundColor:"rgba(192,192,75,0.4)",borderColor:"rgba(192,192,75,1)",borderCapStyle:"butt",borderDash:[],borderDashOffset:0,borderJoinStyle:"miter",pointBorderColor:"rgba(192,192,75,1)",pointBackgroundColor:"#fff",pointBorderWidth:1,pointHoverRadius:5,pointHoverBackgroundColor:"rgba(192,192,75,1)",pointHoverBorderColor:"rgba(220,220,220,1)",pointHoverBorderWidth:2,pointRadius:1,pointHitRadius:10,data:s}]};this.setState({chart_data1:j}),this.setState({chart_data2:w}),this.setState({chart_data3:O}),this.setState({chart_ready:!0})}},{key:"switch_button",value:function(){!0===this.state.button_mode?(this.setState({button_mode:!1}),this.compose_chart()):this.setState({button_mode:!0})}},{key:"increment_tap1",value:function(){if(1!==this.state.last_pressed){var t=(Date.now()-this.state.startAt)/1e3;this.setState({last_pressed:1}),this.setState({dist_array:[].concat(Object(c.a)(this.state.dist_array),[1])}),this.setState({dist_time_array:[].concat(Object(c.a)(this.state.dist_time_array),[t])}),this.setState({tap_count:[].concat(Object(c.a)(this.state.tap_count),[t])})}}},{key:"increment_tap2",value:function(){if(2!==this.state.last_pressed){var t=(Date.now()-this.state.startAt)/1e3;this.setState({last_pressed:2}),this.setState({dist_array:[].concat(Object(c.a)(this.state.dist_array),[1])}),this.setState({dist_time_array:[].concat(Object(c.a)(this.state.dist_time_array),[t])}),this.setState({tap_count:[].concat(Object(c.a)(this.state.tap_count),[t])})}}},{key:"norm",value:function(t,e){return Object(P.b)(Object(P.a)(t[0]-e[0],2)+Object(P.a)(t[1]-e[1],2))}},{key:"reset_counter",value:function(){clearInterval(this.state.ID),null!=this.state.ctx&&this.state.ctx.clearRect(0,0,this.canvasRef.current.width,this.canvasRef.current.height),this.setState({ID:0,tap_count:[],rotate_count:[],fist_count:[],index_passed:0,rotate_passed:0,last_pressed:0,real_time_inferencing:!1,recording:!1,button_mode:!1,chart_ready:!1,finger_done:!1,rotate_done:!1,fist_done:!1,dist_array:[],dist_time_array:[],dist_record:[],dist_time_record:[],rotate_array:[],rotate_time_array:[],rotate_record:[],rotate_time_record:[],fist_array:[],fist_time_array:[],fist_record:[],fist_time_record:[],gait_record:[],gait_time_record:[],chart_data1:null,chart_data2:null,chart_data3:null,wait:!1,wait_till:0,startAt:Date.now(),dead_frame:0})}},{key:"runHandpose",value:function(){var t=Object(_.a)(h.a.mark((function t(){var e,r,i,s=this;return h.a.wrap((function(t){for(;;)switch(t.prev=t.next){case 0:return e=this.canvasRef.current.getContext("2d"),k(e,{text:"Loading",x:180,y:70}),a(162),t.next=5,g.a();case 5:r=t.sent,e.clearRect(0,0,this.canvasRef.current.width,this.canvasRef.current.height),console.log("Handpose model loaded."),this.setState({startAt:Date.now()}),i=setInterval((function(){s.read_time_inference(r)}),50),this.setState({ID:i}),this.setState({real_time_inferencing:!0});case 12:case"end":return t.stop()}}),t,this)})));return function(){return t.apply(this,arguments)}}()},{key:"read_time_inference",value:function(){var t=Object(_.a)(h.a.mark((function t(e){var a,r,i,s,n,o=this;return h.a.wrap((function(t){for(;;)switch(t.prev=t.next){case 0:if(this.state.wait&&(this.setState({wait_till:Date.now()+3e3}),this.setState({wait:!1})),"undefined"===typeof this.webcamRef.current||null===this.webcamRef.current||4!==this.webcamRef.current.video.readyState){t.next=16;break}return a=this.webcamRef.current.video,r=this.webcamRef.current.video.videoWidth,i=this.webcamRef.current.video.videoHeight,this.webcamRef.current.video.width=r,this.webcamRef.current.video.height=i,this.canvasRef.current.width=r,this.canvasRef.current.height=i,t.next=11,e.estimateHands(a);case 11:s=t.sent,n=this.canvasRef.current.getContext("2d"),this.setState({ctx:n}),s.length>0&&R(s,n),Date.now()<this.state.wait_till?(console.log("Waiting till ",this.state.wait_till),this.state.wait_till-Date.now()<1e3?k(n,{text:"1",x:180,y:70}):this.state.wait_till-Date.now()<2e3?k(n,{text:"2",x:180,y:70}):this.state.wait_till-Date.now()<3e3&&k(n,{text:"3",x:180,y:70})):s.length>0?s.forEach((function(t){o.setState({dead_frame:0});var e=t.landmarks,a=o.norm(e[0],e[17]),r=(Date.now()-o.state.startAt)/1e3;if(!1===o.state.finger_done){var i=o.norm(e[4],e[8]);i/=a,o.setState({pawn_dist_array:[].concat(Object(c.a)(o.state.pawn_dist_array),[a])}),o.setState({dist_array:[].concat(Object(c.a)(o.state.dist_array),[i])}),o.setState({dist_time_array:[].concat(Object(c.a)(o.state.dist_time_array),[r])});i>=.52&&o.setState({index_passed:1}),i<.5&&1===o.state.index_passed&&(o.setState({index_passed:0}),o.setState({tap_count:[].concat(Object(c.a)(o.state.tap_count),[r])}))}if(!0===o.state.finger_done&&!1===o.state.rotate_done){var s=(e[2][0]-e[17][0])/a;o.setState({pawn_rotate_array:[].concat(Object(c.a)(o.state.pawn_rotate_array),[a])}),o.setState({rotate_array:[].concat(Object(c.a)(o.state.rotate_array),[s])}),o.setState({rotate_time_array:[].concat(Object(c.a)(o.state.rotate_time_array),[r])}),0===o.state.rotate_passed&&(s>=.25&&o.setState({rotate_passed:1}),s<=-.25&&o.setState({rotate_passed:-1})),1===o.state.rotate_passed&&s<=-.5&&o.setState({rotate_passed:-1}),-1===o.state.rotate_passed&&s>=.5&&(o.setState({rotate_passed:1}),o.setState({rotate_count:[].concat(Object(c.a)(o.state.rotate_count),[r])}))}if(!0===o.state.rotate_done&&!1===o.state.fist_done){var n=(e[8][1]-e[5][1]+(e[12][1]-e[9][1])+(e[16][1]-e[13][1])+(e[20][1]-e[17][1]))/(4*a);o.setState({pawn_fist_array:[].concat(Object(c.a)(o.state.pawn_fist_array),[a])}),o.setState({fist_array:[].concat(Object(c.a)(o.state.fist_array),[n])}),o.setState({fist_time_array:[].concat(Object(c.a)(o.state.fist_time_array),[r])}),n>=0&&o.setState({fist_passed:1}),n<-.4&&1===o.state.fist_passed&&(o.setState({fist_passed:0}),o.setState({fist_count:[].concat(Object(c.a)(o.state.fist_count),[r])}))}})):(this.state.dead_frame>9&&k(n,{text:"Hand Off Screen",x:180,y:70}),this.setState({dead_frame:this.state.dead_frame+1}));case 16:case"end":return t.stop()}}),t,this)})));return function(e){return t.apply(this,arguments)}}()},{key:"stop_tapping",value:function(){var t=Object(_.a)(h.a.mark((function t(){return h.a.wrap((function(t){for(;;)switch(t.prev=t.next){case 0:this.setState({finger_done:!0}),this.setState({wait:!0});case 2:case"end":return t.stop()}}),t,this)})));return function(){return t.apply(this,arguments)}}()},{key:"stop_rotating",value:function(){var t=Object(_.a)(h.a.mark((function t(){return h.a.wrap((function(t){for(;;)switch(t.prev=t.next){case 0:this.setState({rotate_done:!0}),this.setState({wait:!0});case 2:case"end":return t.stop()}}),t,this)})));return function(){return t.apply(this,arguments)}}()},{key:"stop_real_time_inference",value:function(){clearInterval(this.state.ID),this.setState({real_time_inferencing:!1,finger_done:!1,rotate_done:!1,fist_done:!1}),this.compose_chart()}},{key:"record_video",value:function(){var t=Object(_.a)(h.a.mark((function t(){var e,a=this;return h.a.wrap((function(t){for(;;)switch(t.prev=t.next){case 0:this.setState({startAt:Date.now()}),console.log("Handpose model loaded."),e=setInterval((function(){a.concat_frame()}),50),this.setState({ID:e}),this.setState({recording:!0});case 5:case"end":return t.stop()}}),t,this)})));return function(){return t.apply(this,arguments)}}()},{key:"concat_frame",value:function(){var t=Object(_.a)(h.a.mark((function t(){var e,a,r,i;return h.a.wrap((function(t){for(;;)switch(t.prev=t.next){case 0:this.state.wait&&(this.setState({wait_till:Date.now()+3e3}),this.setState({wait:!1})),"undefined"!==typeof this.webcamRef.current&&null!==this.webcamRef.current&&4===this.webcamRef.current.video.readyState&&(Date.now()<this.state.wait_till?(console.log("Waiting till ",this.state.wait_till),e=this.canvasRef.current.getContext("2d"),this.state.wait_till-Date.now()<1e3?k(e,{text:"1",x:180,y:70}):this.state.wait_till-Date.now()<2e3?k(e,{text:"2",x:180,y:70}):this.state.wait_till-Date.now()<3e3&&k(e,{text:"3",x:180,y:70})):(a=(Date.now()-this.state.startAt)/1e3,r=this.webcamRef.current.getScreenshot(),(i=document.createElement("img")).src=r,i.onload=function(){!1===this.state.finger_done?(this.setState({dist_record:[].concat(Object(c.a)(this.state.dist_record),[i])}),this.setState({dist_time_record:[].concat(Object(c.a)(this.state.dist_time_record),[a])})):!0===this.state.finger_done&&!1===this.state.rotate_done?(this.setState({rotate_record:[].concat(Object(c.a)(this.state.rotate_record),[i])}),this.setState({rotate_time_record:[].concat(Object(c.a)(this.state.rotate_time_record),[a])})):!0===this.state.rotate_done&&!1===this.state.fist_done&&(this.setState({fist_record:[].concat(Object(c.a)(this.state.fist_record),[i])}),this.setState({fist_time_record:[].concat(Object(c.a)(this.state.fist_time_record),[a])}))}.bind(this)));case 2:case"end":return t.stop()}}),t,this)})));return function(){return t.apply(this,arguments)}}()},{key:"inference",value:function(){var t=Object(_.a)(h.a.mark((function t(){var e,r,i,s,n,o,d,_=this;return h.a.wrap((function(t){for(;;)switch(t.prev=t.next){case 0:return a(162),t.next=3,g.a();case 3:e=t.sent,console.log("Handpose model loaded."),r=h.a.mark((function t(a){var r;return h.a.wrap((function(t){for(;;)switch(t.prev=t.next){case 0:return t.next=2,e.estimateHands(_.state.dist_record[a]);case 2:(r=t.sent).length>0&&r.forEach((function(t){var e=t.landmarks,r=_.norm(e[0],e[17]),i=_.norm(e[4],e[8]);i/=r;_.setState({pawn_dist_array:[].concat(Object(c.a)(_.state.pawn_dist_array),[r])}),_.setState({dist_array:[].concat(Object(c.a)(_.state.dist_array),[i])}),_.setState({dist_time_array:[].concat(Object(c.a)(_.state.dist_time_array),[_.state.dist_time_record[a]])}),i>=.52&&_.setState({index_passed:1}),i<.5&&1===_.state.index_passed&&(_.setState({index_passed:0}),_.setState({tap_count:[].concat(Object(c.a)(_.state.tap_count),[_.state.dist_time_record[a]])})),console.log("INDEX COUNT:",_.state.tap_count)}));case 4:case"end":return t.stop()}}),t)})),i=0;case 7:if(!(i<this.state.dist_record.length)){t.next=12;break}return t.delegateYield(r(i),"t0",9);case 9:i++,t.next=7;break;case 12:s=h.a.mark((function t(a){var r;return h.a.wrap((function(t){for(;;)switch(t.prev=t.next){case 0:return t.next=2,e.estimateHands(_.state.rotate_record[a]);case 2:(r=t.sent).length>0&&r.forEach((function(t){var e=t.landmarks,r=_.norm(e[0],e[17]),i=(e[2][0]-e[17][0])/r;_.setState({pawn_rotate_array:[].concat(Object(c.a)(_.state.pawn_rotate_array),[r])}),_.setState({rotate_array:[].concat(Object(c.a)(_.state.rotate_array),[i])}),_.setState({rotate_time_array:[].concat(Object(c.a)(_.state.rotate_time_array),[_.state.rotate_time_record[a]])}),0===_.state.rotate_passed&&(i>=.5&&_.setState({rotate_passed:1}),i<=-.5&&_.setState({rotate_passed:-1})),1===_.state.rotate_passed&&i<=-.5&&_.setState({rotate_passed:-1}),-1===_.state.rotate_passed&&i>=.5&&(_.setState({rotate_passed:1}),_.setState({rotate_count:[].concat(Object(c.a)(_.state.rotate_count),[_.state.rotate_time_record[a]])})),console.log("ROTATE COUNT:",_.state.rotate_count)}));case 4:case"end":return t.stop()}}),t)})),n=0;case 14:if(!(n<this.state.rotate_record.length)){t.next=19;break}return t.delegateYield(s(n),"t1",16);case 16:n++,t.next=14;break;case 19:o=h.a.mark((function t(a){var r;return h.a.wrap((function(t){for(;;)switch(t.prev=t.next){case 0:return t.next=2,e.estimateHands(_.state.fist_record[a]);case 2:(r=t.sent).length>0&&r.forEach((function(t){var e=t.landmarks,r=_.norm(e[0],e[17]),i=(e[8][1]-e[5][1]+(e[12][1]-e[9][1])+(e[16][1]-e[13][1])+(e[20][1]-e[17][1]))/(4*r);_.setState({pawn_fist_array:[].concat(Object(c.a)(_.state.pawn_fist_array),[r])}),_.setState({fist_array:[].concat(Object(c.a)(_.state.fist_array),[i])}),_.setState({fist_time_array:[].concat(Object(c.a)(_.state.fist_time_array),[_.state.fist_time_record[a]])}),i>=0&&_.setState({fist_passed:1}),i<-.4&&1===_.state.fist_passed&&(_.setState({fist_passed:0}),_.setState({fist_count:[].concat(Object(c.a)(_.state.fist_count),[_.state.fist_time_record[a]])})),console.log("FIST COUNT:",_.state.fist_count)}));case 4:case"end":return t.stop()}}),t)})),d=0;case 21:if(!(d<this.state.fist_record.length)){t.next=26;break}return t.delegateYield(o(d),"t2",23);case 23:d++,t.next=21;break;case 26:this.setState({record:[]});case 27:case"end":return t.stop()}}),t,this)})));return function(){return t.apply(this,arguments)}}()},{key:"stop_record",value:function(){var t=Object(_.a)(h.a.mark((function t(){return h.a.wrap((function(t){for(;;)switch(t.prev=t.next){case 0:return clearInterval(this.state.ID),t.next=3,this.inference();case 3:this.setState({recording:!1,finger_done:!1,rotate_done:!1,fist_done:!1}),this.compose_chart();case 5:case"end":return t.stop()}}),t,this)})));return function(){return t.apply(this,arguments)}}()},{key:"render",value:function(){var t={facingMode:this.state.facingMode};return Object(r.jsxs)("div",{className:"App",children:[Object(r.jsxs)("header",{className:"App-header",children:[Object(r.jsx)(w.a,{ref:this.webcamRef,videoConstraints:t,style:{position:"absolute",marginLeft:"auto",marginRight:"auto",left:0,right:0,textAlign:"center",zindex:9,width:"auto",height:"auto"}}),Object(r.jsx)("canvas",{ref:this.canvasRef,style:{position:"absolute",marginLeft:"auto",marginRight:"auto",left:0,right:0,textAlign:"center",zindex:9,width:"auto",height:"auto"}})]}),this.state.button_mode?Object(r.jsxs)("div",{children:[Object(r.jsx)("button",{onClick:this.increment_tap1,id:"button1",children:"Index Finger"}),Object(r.jsx)("button",{onClick:this.increment_tap2,id:"button2",children:"Middle Finger"})]}):Object(r.jsx)("div",{}),Object(r.jsxs)("div",{children:[this.state.real_time_inferencing?this.state.finger_done?this.state.rotate_done?Object(r.jsx)(m.a,{variant:"contained",color:"primary",onClick:this.stop_real_time_inference,children:"Calculate Result"}):Object(r.jsx)(m.a,{variant:"contained",color:"primary",onClick:this.stop_rotating,children:"Finish Rotating"}):Object(r.jsx)(m.a,{variant:"contained",color:"primary",onClick:this.stop_tapping,children:"Finish Tapping"}):Object(r.jsx)(m.a,{disabled:this.state.recording,variant:"contained",color:"primary",onClick:this.runHandpose,children:"Starting Real Time Inference"}),this.state.recording?this.state.finger_done?this.state.rotate_done?Object(r.jsx)(m.a,{variant:"contained",color:"secondary",onClick:this.stop_record,children:"Calculate Result"}):Object(r.jsx)(m.a,{variant:"contained",color:"secondary",onClick:this.stop_rotating,children:"Finish Rotating"}):Object(r.jsx)(m.a,{variant:"contained",color:"secondary",onClick:this.stop_tapping,children:"Finish Tapping"}):Object(r.jsx)(m.a,{disabled:this.state.real_time_inferencing,variant:"contained",color:"secondary",onClick:this.record_video,children:"Starting Recording"}),Object(r.jsx)(m.a,{disabled:this.state.recording||this.state.real_time_inferencing,variant:"outlined",color:"secondary",onClick:this.reset_counter,children:"Reset All"})]}),Object(r.jsx)("input",{type:"number",id:"real_measurement",onChange:this.compose_chart,step:"0.001",min:"0",max:"20"}),Object(r.jsx)("button",{disabled:!this.state.chart_ready,onClick:this.switch_style,children:"Switch Chart Style"}),Object(r.jsx)("button",{onClick:this.switch_cam,children:"Switch Camera"}),Object(r.jsx)("button",{disabled:this.state.recording||this.state.real_time_inferencing,onClick:this.runPosenet,children:"PoseNet"}),Object(r.jsx)("button",{disabled:this.state.recording||this.state.real_time_inferencing,onClick:this.runFacemesh,children:"Facemesh"}),Object(r.jsx)("button",{disabled:this.state.recording||this.state.real_time_inferencing,onClick:this.switch_button,children:"Switch On/Off Button"}),Object(r.jsxs)("h5",{children:["Finger Tapping Count:",this.state.tap_count.length,"\xa0\xa0\xa0\xa0 Rotate Count:",this.state.rotate_count.length,"\xa0\xa0\xa0\xa0 Gripping Count: ",this.state.fist_count.length]}),Object(r.jsxs)("div",{children:[this.state.chart_ready?Object(r.jsxs)("div",{children:[Object(r.jsx)(A.Line,{data:this.state.chart_data1}),Object(r.jsx)(A.Line,{data:this.state.chart_data2}),Object(r.jsx)(A.Line,{data:this.state.chart_data3})]}):Object(r.jsx)("div",{}),Object(r.jsx)("div",{children:Object(r.jsx)("input",{type:"file",id:"upload-json"})}),Object(r.jsx)("div",{children:Object(r.jsx)("input",{type:"file",id:"upload-weights"})})]})]})}}]),i}(s.a.Component),W=function(t){t&&t instanceof Function&&a.e(3).then(a.bind(null,434)).then((function(e){var a=e.getCLS,r=e.getFID,i=e.getFCP,s=e.getLCP,n=e.getTTFB;a(t),r(t),i(t),s(t),n(t)}))};o.a.render(Object(r.jsx)(s.a.StrictMode,{children:Object(r.jsx)(M,{})}),document.getElementById("root")),W()}},[[415,1,2]]]);
//# sourceMappingURL=main.32f129b9.chunk.js.map