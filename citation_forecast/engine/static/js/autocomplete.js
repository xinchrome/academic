
var auto_divnode_style = "background:rgba(245,245,245,1);min-height:20px;min-width:30px;padding:5px;"
var highlightindex = -1;//表示当前高亮节点
var timeoutId;

function lightEventHide(){
  var comText = $("#auto").hide().children("div").eq(highlightindex).text();
  $("#user_search").val(comText);
}
function lightEvent(){
  var comText = $("#auto").children("div").eq(highlightindex).text();
  $("#user_search").val(comText);
}
function autoHide(){
  $("#auto").hide();
}

$(document).ready(function() {
    var wordInput = $("#user_search");//文本框
    var wordInputOffset = wordInput.offset();//获得文本框位置
    $("#auto").hide().css("border", "1px black solid;").css("position", "absolute")
        .css("top", wordInputOffset.top + wordInput.height() + 5 + "px")
        .css("left", wordInputOffset.left + "px").width(wordInput.width() + 3 + "px")
        .css("-webkit-box-shadow","rgba(0, 0, 0, 0.5) 0px 0px 5px")
        .css("-moz-box-shadow","rgba(0, 0, 0, 0.5) 0px 0px 5px")
        .css("box-shadow","rgba(0, 0, 0, 0.5) 0px 0px 2px")
        .css("font","100% Helvetica Neue, Helvetica, Arial, sans-serif")
        .css("backgroundColor","white");
    wordInput.keyup(function(event) {
        //处理文本框中的键盘事件
        //如果输入字母，将文本框中最新信息发送给服务器
        var myEvent = event || window.event;
        var keyCode = myEvent.keyCode;//获得键值

        if (keyCode == 27) {
          var wordText = $("#user_search").val();
          autoHide();
          wordInput.text(wordText);
        }
        else {
          if (keyCode >= 65 && keyCode <= 90 || keyCode == 8 || keyCode == 46) { //8对应退格键，46对应删除键
            var wordText = $("#user_search").val();//获得文本框中的内容
            var autoNode = document.getElementById("auto"); //$("#auto");
            autoNode.innerHTML = "";
            if (wordText != "") {
              clearTimeout(timeoutId);//对上次未完成的延时操作进行取消
              //延时操作，减少与服务器的交互次数，延时500ms，防止用户操作过快
              timeoutId = setTimeout(function() {
                $.getJSON("autocomplete/", {prefix:wordText}, function(data) {//发送数据，第二项是属性名对应属性值
                    //var jqueryObj = $(data);//将dom对象data转换成jQuery的对象
                    var wordNodes = data.title;//find("title");//找到所有word节点
                    console.log(wordNodes);
                    autoNode.innerHTML = "";
                    //wordNodes.each(function(i) { //i是索引，用来给id赋值
                    for (var i = 0;i < wordNodes.length;i++){
                        var wordNode = $(this);//获取单词内容
                        var newDivNode = document.createElement("div"); //$("<div>").attr("id", i).css("backgroundColor", "white");
                        newDivNode.id = i;
                        newDivNode.style = auto_divnode_style;
                        newDivNode.innerText = wordNodes[i].substring(0,60);
                        //newDivNode.html(wordNode.text()).appendTo(autoNode);//新建div节点，加入单词内容
                        $(newDivNode).css("backgroundColor", "white");
                        autoNode.appendChild(newDivNode);
                        //增加鼠标进入事件，高亮节点
                        newDivNode.onmouseover = function(event) {
                          //将原来高亮的节点取消高亮
                          if (highlightindex != -1) {
                            $("#auto").children("div").eq(highlightindex)
                                .css("backgroundColor", "white");
                          }
                          //记录新的高亮索引
                          highlightindex = $(this).attr("id");
                          $(this).css("backgroundColor", "#3366CC").css("cursor","pointer");
                        };
                        //增加鼠标移出事件，取消节点高亮
                        newDivNode.onmouseout = function(event) {
                              //取消鼠标移出节点的高亮
                              $(this).css("backgroundColor", "white");
                        };
                        //增加鼠标点击事件,可以进行补全
                        newDivNode.onclick = function() {
                          //取出高亮节点的文本内容
                          var comText = $(this).text();
                          autoHide();
                          highlightindex = -1;
                          //文本框内容变为高亮节点内容
                          $("#user_search").val(comText);
                        };
                    
                    };
                    //添加单词内容到弹出框
                    console.log("@3");
                    if (wordNodes.length > 0) {
                      $(autoNode).show();
                    } else {
                      $(autoNode).hide();
                      highlightindex = -1;//弹出框隐藏，高亮节点索引设成-1
                    }
                });
              }, 1);
            } else {
              $(autoNode).hide();
              highlightindex = -1;
            }
          } else if (keyCode == 38 || keyCode == 40) {   //判断是否输入的是向上38向下40按键
            if (keyCode == 38) {
              var autoNodes = $("#auto").children("div").css("background-color", "white");
              if (highlightindex != -1) {
                autoNodes.eq(highlightindex).css("background-color", "white");
                highlightindex--;
              } else {
                lightEvent();
                highlightindex = autoNodes.length - 1;
              }
              if (highlightindex == -1) {
                highlightindex = autoNodes.length - 1;//如果改变索引值后index变成-1，则将索引值指向最后一个元素
              }
              lightEvent();
              autoNodes.eq(highlightindex).css("backgroundColor", "#3366CC");
            }
            if (keyCode == 40) {
              var autoNodes = $("#auto").children("div");
              if (highlightindex != -1) {
                autoNodes.eq(highlightindex).css("background-color", "white");
              }
              highlightindex++;
              if (highlightindex == autoNodes.length) {
                highlightindex = 0;//如果改变索引值等于最大长度，则将索引值指向第一个元素

              }
              lightEvent();
              autoNodes.eq(highlightindex).css("backgroundColor", "#3366CC");
            }
          } else if (keyCode == 13) {	   //判断是否按下回车键
            //下拉框有高亮
            if (highlightindex != -1) {
              lightEventHide();
              highlightindex = -1;
            } else {
              //alert("文本框中的[" + $("#user_search").val() + "]被提交了");
              $("#auto").hide();
              $("#user_search").get(0).blur();//让文本框失去焦点
            }
            //下拉框没有高亮
          }
        }
      }
    )
    ;
    $("input[type='button']").click(function() {
      alert("文本框中的[" + $("#user_search").val() + "]被提交了");
    });
});