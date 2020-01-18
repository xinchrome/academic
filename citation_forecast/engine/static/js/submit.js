
//=============================style=============================================
var paper_div_style = "margin: 0px 0px 10px 0px;  padding: 5px;background: rgba(245, 245, 245, 1);\
min-height: 70px;min-width: 30px;border: 1px solid #BBB;-webkit-box-shadow: rgba(0, 0, 0, 0.5) 0px 0px 5px;\
  -moz-box-shadow: rgba(0, 0, 0, 0.5) 0px 0px 5px;box-shadow: rgba(0, 0, 0, 0.5) 0px 0px 2px;\
  font: 100% Helvetica Neue, Helvetica, Arial, sans-serif;";


//=============================draw axis=============================================

var xCenter = 300;	     
var yCenter = 300;	     

var xMargin = 40;
var yMargin = 40;

var rect = [40,44,540,544];

var current_max_y = 50;
var current_max_x = 2025;
var current_min_x = 2010;

function coordinate(){
	var xElem = document.getElementById("CANVAS");
	var cxt = xElem.getContext("2d");

	cxt.moveTo(xMargin,2 * yCenter - yMargin);
	cxt.lineTo(2 * xCenter - xMargin,2 * yCenter - yMargin);

	cxt.strokeStyle = "black"; 
	cxt.stroke();

	cxt.moveTo(xMargin,yMargin);
	cxt.lineTo(xMargin,2 * yCenter - yMargin);

	cxt.strokeStyle = "black"; 
	cxt.stroke();
}

function calibration(){
	var yCalibration = 60;
	var xCalibration = 50;
	var cElem = document.getElementById("CANVAS");
	var cCxt = cElem.getContext("2d");

	for(var i=0; i<11; i++){ 
		cCxt.moveTo(xMargin,yCalibration);
		cCxt.lineTo(xMargin + 7,yCalibration);

		cCxt.strokeStyle = "black"; 
		cCxt.stroke();

		cCxt.moveTo(xCalibration,2 * yCenter - yMargin - 7);
		cCxt.lineTo(xCalibration,2 * yCenter - yMargin);

		cCxt.strokeStyle = "black"; 
		cCxt.stroke();

		yCalibration += 50;
		xCalibration += 50;
	}
}

function arrow(){
	var aElem = document.getElementById("CANVAS");
	var aCxt = aElem.getContext("2d");

	aCxt.moveTo(2 * xCenter - xMargin - 10,2 * yCenter - yMargin - 10);
	aCxt.lineTo(2 * xCenter - xMargin,2 * yCenter - yMargin);
	aCxt.lineTo(2 * xCenter - xMargin - 10,2 * yCenter - yMargin + 10);

	aCxt.strokeStyle = "black"; 
	aCxt.stroke();

	aCxt.moveTo(xMargin - 10,yMargin + 10);
	aCxt.lineTo(xMargin,yMargin);
	aCxt.lineTo(xMargin + 10,yMargin + 10);

	aCxt.strokeStyle = "black"; 
	aCxt.stroke();
}

function legend(start){
	var aElem = document.getElementById("CANVAS");
	var aCxt = aElem.getContext("2d");
	//cCxt.clearRect(xMargin + 20 , yMargin, 40, 60);

	aCxt.strokeStyle = "black";
	//aCxt.strokeText("black",xMargin + 20,yMargin);
	aCxt.strokeText("citation count",xMargin - 35,yMargin - 20);
	aCxt.strokeText("year",2 * xCenter - xMargin + 10,2 * yCenter - yMargin + 30);

}

function addRange(x,o,y,detail_y){
	current_max_y = y;
	current_min_x = o;
	current_max_x = x;
	var aElem = document.getElementById("LINE_CANVAS");
	var aCxt = aElem.getContext("2d");	

	aCxt.strokeStyle = "black";
	aCxt.strokeText(String(Math.floor(y)),xMargin - 25,yMargin);
	aCxt.strokeText(String(Math.floor(o)),xMargin - 25,2 * yCenter - yMargin);
	aCxt.strokeText(String(Math.floor(x)),2 * (xCenter - 20) - xMargin,2 * yCenter - yMargin);
	var gap_x = (x - o)/3;
	var gap_y = y/5;
	for(var i = 1;i < 3;i++){
		aCxt.strokeText(String(Math.floor(o + gap_x * i)),xMargin + i * 2/3 * (xCenter - 20 - xMargin),2 * yCenter - yMargin);		
	}
	for(var i = 1;i < 5;i++){
		aCxt.strokeText(String(Math.floor(y - gap_y * i)),xMargin - 25,yMargin + i * 2/5 * (yCenter - yMargin));
	}
}

function showPoint(x,y){
	var _x = x > rect[0]? (x - rect[0])/(rect[2] - rect[0]) * (current_max_x - current_min_x) + current_min_x : 0;
	var _y = y < rect[3]? (rect[3] - y)/(rect[3] - rect[1]) * current_max_y : 0;
	$("#coordinate_text").html("<span style=\"color:DodgerBlue ;\">year</span>: " + 
		String(_x).substring(0,4) + ", <span style=\"color:DodgerBlue ;\">citation</span>: " + String(Math.floor(_y)));
}

//=============================draw curve=============================================


function addPoint(canvas,x,y,color){
	var physical_x = xMargin + 10 * x;
	var physical_y = 2 * yCenter - 1.5 * yMargin - 10 * y;

	var context = canvas.getContext("2d");
	context.globalAlpha = 0.7;
	context.fillStyle=color;
	context.beginPath();
	context.arc(physical_x, physical_y, 2, 0, Math.PI*2, true);
	context.fill();
}

function addLine(canvas,x1,y1,x2,y2,color){
	var physical_x1 = xMargin + 10 * x1;
	var physical_y1 = 2 * yCenter - 1.5 * yMargin - 10 * y1;

	var physical_x2 = xMargin + 10 * x2;
	var physical_y2 = 2 * yCenter - 1.5 * yMargin - 10 * y2;

	var context = canvas.getContext("2d");
	context.moveTo(physical_x1,physical_y1);
	context.lineTo(physical_x2,physical_y2);
	context.strokeStyle = color;
	context.stroke();
}

function clean_canvas(){
	var cElem = document.getElementById("LINE_CANVAS");
	var cCxt = cElem.getContext("2d");
	cCxt.clearRect(0 , 0, cElem.width, cElem.height);
}

function draw_one_curve(points,pred_points,color){
	console.log('@2');
	clean_canvas();
	clearInterval(task_id);
	var canvas = document.getElementById("LINE_CANVAS");
	//var color = "green";
	var height_rate = 1;
	// var width_rate = 1;
	var width_unit = (rect[3] - rect[0])/(points.length + pred_points.length - 1)/10;
	var max_y = 0;
	var max_x = points.length;
	for (var j = 0;j < pred_points.length;j++){
		if(pred_points[j] > max_y){
			max_y = pred_points[j];
		}
	}

	if(max_y > 50){
		max_y = Math.floor(max_y/10)*10 + 10;
		height_rate = 50.0/max_y;
	}

	console.log(points);

	var j = 0;
	var split = points.length - 1;
	var task_id = setInterval(function(){
		
		if (j < 1) {
			addPoint(canvas,j * width_unit,points[j] * height_rate,color);
			j++;
		} else if(j < points.length){
			addPoint(canvas,j * width_unit,points[j] * height_rate,color);
			addLine(canvas,(j-1) * width_unit,points[j-1] * height_rate,j * width_unit,points[j] * height_rate,color);
			j++;
		} else if (j < points.length + pred_points.length){
			var jj = j - points.length;
			if (jj < 1) {
				addPoint(canvas,j * width_unit,pred_points[jj] * height_rate,color);
				addLine(canvas,(j-1) * width_unit,points[j-1] * height_rate,j * width_unit,pred_points[jj] * height_rate,color);
				j++;
			} else {
				addPoint(canvas,j * width_unit,pred_points[jj] * height_rate,color);
				addLine(canvas,(j-1) * width_unit,pred_points[jj-1] * height_rate,j * width_unit,pred_points[jj] * height_rate,color);
				j++;
			}
		} else if (j < points.length + pred_points.length + 11) {
			var jj = j - points.length - pred_points.length;
			addPoint(canvas,split * width_unit,points[split] * height_rate + (jj - 5),"black");
			j++;
		} else {
			clearInterval(task_id);
		}
	},10);
	addRange(2025,max_x > 10 ? 2025 - 10 - max_x : 2005,max_y > 50? max_y : 50,false);
}


function draw_curves(points_array,pred_array,color_array){
	clean_canvas();
	if (points_array.length <= 0) {
		return;
	}
	console.log('@1');
	var canvas = document.getElementById("LINE_CANVAS");
	var color = "black";
	var height_rate = 1;
	//var width_rate = 1;
	var max_y = 0
	var max_x = 0;
	for(var i = 0;i < pred_array.length;i++){
		for (var j = 0;j < pred_array[i].length;j++){
			if(pred_array[i][j] > max_y){
				max_y = pred_array[i][j];
			}
		}
		if (points_array[i].length > max_x) {
			max_x = points_array[i].length;
		}
	}
	max_x = max_x + 1;
	var width_unit = (rect[3] - rect[0])/(max_x + pred_array[0].length - 1)/10;

	if(max_y > 50){
		max_y = Math.floor(max_y/10)*10 + 10;
		height_rate = 50.0/max_y;
	}

	var split = 50 - 10 * width_unit;
	var ncurves = points_array.length < color_array.length ? points_array.length : color_array.length;
	for(var i = 0;i < ncurves;i++){
		// if(i==1) color = "blue";
		// if(i==2) color = "green";
		color = color_array[i];
		console.log(points_array[i]);

		for (var j = 0;j < points_array[i].length;j++){
			if (j < 1) {
				addPoint(canvas,split - (points_array[i].length - 1 - j) * width_unit,points_array[i][j] * height_rate,color);
			} else {
				addPoint(canvas,split - (points_array[i].length - 1 - j) * width_unit,points_array[i][j] * height_rate,color);
				addLine(canvas,split - (points_array[i].length - j) * width_unit,points_array[i][j-1] * height_rate,
					split - (points_array[i].length - 1 - j) * width_unit,points_array[i][j] * height_rate,color);
			}
		}
		for (var j = 0;j < pred_array[i].length;j++){
			if (j < 1) {
				addPoint(canvas,(j + 1) * width_unit + split,pred_array[i][j] * height_rate,color);
				addLine(canvas,j * width_unit + split,points_array[i][points_array[i].length-1] * height_rate,
					(j + 1) * width_unit + split,pred_array[i][j] * height_rate,color);
			} else {
				addPoint(canvas,(j + 1) * width_unit + split,pred_array[i][j] * height_rate,color);
				addLine(canvas,j * width_unit + split,pred_array[i][j-1] * height_rate,
					(j + 1) * width_unit + split,pred_array[i][j] * height_rate,color);
			}
		}
	}
	for (var i = 0;i < 50;i++) {
		addPoint(canvas,split,i,"black");
	}
	addRange(2025,max_x > 10 ? 2025 - 10 - max_x : 2005,max_y > 50? max_y : 50,true);
}


//=============================event handling=============================================

var points_array = new Array();
var pred_array = new Array();
var colors = ["red","blue","green","purple","orange"];
var check_ids = ["checkbox-10-0","checkbox-10-1","checkbox-10-2","checkbox-10-3","checkbox-10-4"];

function onclick_title(title_id){
	var index = Number(title_id.split('_').pop());
	console.log(index);
	for (var i=0;i < check_ids.length && i < points_array.length;i++) {
		var check = document.getElementById(check_ids[i]);
		if (i == index) {
			check.checked = true;
		} else {
			check.checked = false;
		}
	}
	draw_one_curve(points_array[index],pred_array[index],colors[index]);
}

function onclick_check(check_id){
	var _points_array = new Array();
	var _pred_array = new Array();
	var _colors = new Array();
	for (var i=0;i < check_ids.length && i < points_array.length;i++) {
		var check = document.getElementById(check_ids[i]);
		if (check.checked == true) {
			_points_array.push(points_array[i]);
			_pred_array.push(pred_array[i]);
			_colors.push(colors[i]);
		}
	}
	draw_curves(_points_array,_pred_array,_colors);
}

function reload_papers(data){
	points_array.length = 0;
	pred_array.length = 0;

	if(data.error != 0){
		var papers = document.getElementById("papers");
		var p = document.createElement("p");
		p.innerText = "No More Related Results";
		papers.innerHTML = "";
		papers.appendChild(document.createElement("br"));
		papers.appendChild(p);
		return -1;
	}

	var title_array = new Array();
	var url_array = new Array();
	// var author_array = new Array();
	var year_array = new Array();
	var journal_array = new Array();
	var conference_array = new Array();
	var keyword_array = new Array();
	var authors_array = new Array();
	var rank_array = new Array();
	var ct_array = new Array();

    $.each(data.paper_list, function(index, paper){
          points_array.push(paper.real_ct);
          pred_array.push(paper.pred_ct);
          title_array.push(paper.title);
          url_array.push(paper.url);
          // author_array.push(paper.author);
          year_array.push(paper.publish_year);

		  journal_array.push(paper.journal);
		  conference_array.push(paper.conference_series);
		  keyword_array.push(paper.keyword);
		  authors_array.push(paper.authors);
		  rank_array.push(paper.paper_rank);
		  ct_array.push(paper.real_ct[paper.real_ct.length-1])
    });
	//$("#predict_text").text(html);
	//$("#predict_text").show();
	console.log(points_array);
	draw_curves(points_array.slice(0,3),pred_array.slice(0,3),[colors[0],colors[1],colors[2]]);
	//$("#paper_titles").text(title_array[0])
	var papers = document.getElementById("papers");
	papers.innerHTML = "";
	var hits = document.createElement("h6");
	hits.innerText = "About " + data.hits + " papers found";
	papers.appendChild(hits);	
	papers.appendChild(document.createElement("p"));	

	var real_start = data.start;
	for (var i = 0;i < title_array.length;i++){
		var div = document.createElement("div");
		div.class = "paper";
		div.style = paper_div_style;

		var check_div = document.createElement("div");
		var html = "<input type=\"checkbox\" id=\"checkbox-10-";
		html += String(i);
		html += "\"/><label for=\"checkbox-10-";
		html += String(i);
		html += "\"></label>";
		check_div.innerHTML = html;
		check_div.style = "float:left;"
		div.appendChild(check_div);

		var title = document.createElement("h5");
		title.id = 'paper_' + i;
		title.onclick = function(){onclick_title(this.id);};
		title.innerText = (real_start + i + 1) + '.  ' + title_array[i];
		title.style = "cursor:pointer;min-height:25px;";
		div.appendChild(title);

		var authors = document.createElement("h6");
		authors.innerHTML = "<span style=\"color:Peru;\">Authors</span> : &nbsp;&nbsp;" + authors_array[i].join(' · ');
		div.appendChild(authors);

		var year = document.createElement("h6");
		var journal = journal_array[i];
		if (journal.length > 0){
			year.innerHTML = "<span style=\"color:Peru;\">Journal</span> : &nbsp;&nbsp;&nbsp;" + journal + " · " + year_array[i];
		} else {
			year.innerHTML = "<span style=\"color:Peru;\">Journal</span> : &nbsp;&nbsp;&nbsp;" + conference_array[i] + " · " + year_array[i];
		}
		div.appendChild(year);

		var keyword = document.createElement("h6");
		keyword.innerHTML = "<span style=\"color:Peru;\">Keyword</span> : " + keyword_array[i];
		div.appendChild(keyword);

		var anchor = document.createElement("a");
		anchor.style.float = "right";
		anchor.href = url_array[i];
		anchor.target = "_blank";
		anchor.innerHTML = "full text >>"
		div.appendChild(anchor);

		var citation = document.createElement("h6");
		citation.innerHTML = "<span style=\"color:Peru;\">Citation</span> : &nbsp;&nbsp;&nbsp;" + ct_array[i];
		div.appendChild(citation);

		papers.appendChild(div);
	}

	setTimeout(function(){
		for (var i = 0;i < check_ids.length && i < points_array.length;i++) {
			var check = document.getElementById(check_ids[i]);
			check.onclick = function(){onclick_check(this.id);};
			if (i < 3) {
				check.checked = true;			
			} else {
				check.checked = false;
			}
		}
	},1);
	//papers.appendChild(document.createElement("p"));
	return real_start;	
}

function perform_search_request(kw,start){
	$("#auto").hide();
	$("#loading_gif").show();
	$.ajax({
		type: "GET",
		url: "submit/",
		data: {"kw":kw,"start":start},
		dataType: "json",
		success: function(data){
			$("#loading_gif").hide();
			var real_start = reload_papers(data);
			if (real_start >= 0){
				$("#current_page").val(parseInt(real_start/5) + 1);
			} else {
				$("#current_page").val("");
			}
		}
	})	
}



$(document).ready(function(){

	$("#loading_gif").hide();

	$("#submit").click(function(){
		var search_text = $("#user_search").val();
		var search_words = $.trim(search_text).split(/\s+/gi);
		console.log(search_words);
		if (search_words.length == 0){
			//alert("You should input id of paper");
			return;
		}

		perform_search_request(search_words,0);
	});

	$("#user_search").keyup(function(event){
		var myEvent = event || window.event;
		var keyCode = myEvent.keyCode;

		if (keyCode == 13) {
			var search_text = $("#user_search").val();
			var search_words = $.trim(search_text).split(/\s+/gi);
			console.log(search_words);
			if (search_words.length == 0){
				//alert("You should input id of paper");
				return;
			}
			perform_search_request(search_words,0);
		}
	});

	$("#current_page").keyup(function(event){
		var myEvent = event || window.event;
		var keyCode = myEvent.keyCode;

		if (keyCode == 13) {
			var page = $("#current_page").val();
			if(isNaN(page)){
				alert("You should input a digital page number");
				return false;
			}
			var start = (parseInt(page) - 1) * 5;
			var search_text = $("#user_search").val();
			var search_words = $.trim(search_text).split(/\s+/gi);
			console.log(search_words);
			if (search_words.length == 0){
				//alert("You should input id of paper");
				return;
			}
			perform_search_request(search_words,start);
		}
	});

	$("#next_page").click(function(){
		var search_text = $("#user_search").val();
		var page = $("#current_page").val();
		if(isNaN(page)){
			alert("You should input a digital page number");
			return false;
		}
		var start = (parseInt(page)) * 5;
		var search_words = $.trim(search_text).split(/\s+/gi);
		console.log(search_words);
		if (search_words.length == 0){
			//alert("You should input id of paper");
			return;
		}

		perform_search_request(search_words,start);
	});

	$("#prev_page").click(function(){
		var search_text = $("#user_search").val();
		var page = $("#current_page").val();
		if(isNaN(page)){
			alert("You should input a digital page number");
			return false;
		}
		var start = (parseInt(page) - 2) * 5;
		var search_words = $.trim(search_text).split(/\s+/gi);
		console.log(search_words);
		if (search_words.length == 0){
			//alert("You should input id of paper");
			return;
		}

		perform_search_request(search_words,start);
	});

	$("#Inspector").click(function(){
		$("#points").slideToggle("normal");
	});

	$("#Searched_papers").click(function(){
		$("#papers").slideToggle("normal");
	});	

});

window.onload = function(){

	coordinate();
	calibration();
	arrow();
	legend(1);
	addRange(2025,2010,50,true);

	var canvas = document.getElementById("LINE_CANVAS");
	canvas.onmousemove = function(event){
		// var x = ( event.pageX - xMargin)/50.0;
		// var y = (2 * yCenter - yMargin - event.pageY)/10.0 + 20.0;

		showPoint(event.pageX - this.offsetLeft,event.pageY - this.offsetTop);
	};

};