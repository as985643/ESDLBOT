
function mapbtn(){
	console.log("Click!");
    $body.addClass("loading");
    $.ajax({
      url: '/navigation/index',
      type: 'POST',
      success: function(response) {
        console.log(response);

        function connect() {

          var ros = new ROSLIB.Ros({
            url: 'ws://192.168.1.180:9090'
          });



          ros.on('connection', function() {
            console.log('Connected to websocket server.');
            var rosTopic = new ROSLIB.Topic({
              ros: ros,
              name: '/rosout',
              messageType: 'rosgraph_msgs/Log'
            });


            rosTopic.subscribe(function(message) {

              //if (message.msg == "Initialization complete") {
              if (true){
                //console.log(message.msg)
                window.location = "/mapping";
                $body.removeClass("loading");
                rosTopic.unsubscribe();
              }

            });

          });


          ros.on('close', function() {
            console.log('Connection to websocket server closed.');

          });


          ros.on('error', function(error) {
            console.log('Error connecting to websocket server: ', error);
            setTimeout(function() {
                connect();
            }, 1000);

        });
      }

      connect();



      //  setTimeout(function(){ window.location ="mapping";
      // $body.removeClass("loading"); }, 10000);

    },
    error: function(error) {
      console.log(error);
    }
  })
};

$(document).ready(function() {

    $body = $("body");

    $(".startmap").click(function(event) {
    	console.log("click");
        $body.addClass("loading");
        event.preventDefault();
        $.ajax({
            url: '/navigation/gotomapping',
            type: 'POST',
            success: function(response) {

                setTimeout(function() {
                    window.location = "/mapping";
                    $body.removeClass("loading");
                }, 5000);

            },
            error: function(error) {
                console.log(error);
            }
        })

    });
    $("#index-list").click(function(event) {

        document.cookie =  `map=${event.target.innerHTML}`;
        $('#exampleModal').modal('hide');
        $.ajax({
            url: '/mapping/cutmapping',
            type: 'POST',
            success: function(response) {

                $.ajax({
                    url: '/index/navigation-precheck',
                    type: 'GET',
                    success: function(response) {
                        console.log(response.mapcount);
                        if (response.mapcount > 0) {
                            $body.addClass("loading");
                            $.ajax({
                                url: '/index/gotonavigation',
                                type: 'POST',
                                data: event.target.innerHTML,
                                success: function(response) {

                                    function connect() {

                               

                                // Connecting to ROS
                                // -----------------
                                var ros = new ROSLIB.Ros({
                                    url: 'ws://192.168.1.180:9090'
                                });


                                console.log('Wait for Connection to websocket server.');
                                ros.on('connection', function() {
                                    console.log('Connected to websocket server.');
                                    var rosTopic = new ROSLIB.Topic({
                                        ros: ros,
                                        name: '/rosout_agg',
                                        messageType: 'rosgraph_msgs/Log'
                                    });


                                    rosTopic.subscribe(function(message) {

                                        //if (message.msg == "odom received!") {
                                        if (true){
                                            //console.log(message.msg)
                                            window.location = "/navigation";
                                            $body.removeClass("loading");
                                            rosTopic.unsubscribe();
                                        }
                                        else alert("odom not received!")

                                    });

                                });


                                ros.on('close', function() {
                                    console.log('Connection to websocket server closed.');

                                });


                                ros.on('error', function(error) {
                                    console.log('Error connecting to websocket server: ', error);
                                    setTimeout(function() {
                                        connect();
                                    }, 1000);

                                });
                            }

                            connect();


                        },
                                error: function(error) {
                                    console.log(error);
                                }

                            })



                        } else {
                            alert("No map in directory.Please do mapping.")
                        }
                    },
                    error: function(error) {
                        console.log(error);
                    }

                })

            },
            error: function(error) {
                console.log(error);
            }

        })

    });


});
