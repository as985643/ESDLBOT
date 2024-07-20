var navigation = false;
var pathed = false;
var homing = false;
var MAP_WIDTH = (window.innerWidth)*0.65;
var MAP_HEIGHT = window.innerHeight - (window.innerHeight)*0.08;
var value = $.cookie("map");
var path = true;


$("#drop-text").change(function() {
    alert($("#drop-text :selected").text())
});
$(document).ready(function() {
    $body = $("body");
    $(".dropdown-content").click(function(event) {
        $body = $("body");
        event.preventDefault();
        robotvisible =true;


        value = event.target.firstChild.data;
	document.cookie =  `map=${value}`;
        imageurl(value);
        $.ajax({
            url: '/navigation/loadmap',
            type: 'POST',
            data: value,
            success: function(response) {
                console.log(response);
            },
            error: function(error) {
                console.log(error);
            }

        })
        $body.addClass("loading");
        loading();

    });

    function loading() {
        var rosTopic = new ROSLIB.Topic({
            ros: ros,
            name: '/rosout_agg',
            messageType: 'rosgraph_msgs/Log'
        });

        rosTopic.subscribe(function(message) {

            if (message.msg == "global_costmap: Pre-Hydro parameter \"static_map\" unused since \"plugins\" is provided") {
                console.log(message.msg)
                $body.removeClass("loading");
                location.reload()
            }

        });
    }

    window.imageurl = function(value) {

        NAV2D.ImageMapClientNav({
            ros: ros,
            viewer: viewer,
            rootObject: viewer.scene,
            serverName: '/move_base',
            image: `/static/maps/${value}.png`
        });
    }



    var ip = $.cookie("serverip");
	
    // Connecting to ROS
    // -----------------
    // let cookies = document.cookie.split('; ');
    // let ipCookie = cookies.find(row => row.startsWith('serverip='));
    // let ip = ipCookie.split('=')[1];
    var ros = new ROSLIB.Ros({
        // url: 'ws://192.168.1.180:9090'
        url: 'ws://' + ip + ':9090'
    });


    // Create the main viewer.
    var viewer = new ROS2D.Viewer({
        divID: 'nav',
        width: MAP_WIDTH,
        height: MAP_HEIGHT
    });



    var imageMapClientNav = new NAV2D.ImageMapClientNav({
        ros: ros,
        viewer: viewer,
        rootObject: viewer.scene,
        serverName: '/move_base',
        image: `/static/maps/${value}.png`
    });




    gridClient = new NAV2D.OccupancyGridClientNav({
        ros: ros,
        rootObject: viewer.scene,
        viewer: viewer,
        serverName: '/move_base',
        continuous: true
    });

    function mapchangegraphical() {

        imageMapClientNav.addImg();
    };


    function mapchangelive() {

        imageMapClientNav.removeImg();

    }
    var pan = new ROS2D.PanView({
        ros: ros,
        rootObject: viewer.scene
    });

    window.pane = function(a, b) {
        pan.startPan(a, b);
    }

    window.paned = function(c, d) {
        pan.pan(c, d);
    }

    window.zoomInMap = function(ros, viewer) {
        var zoom = new ROS2D.ZoomView({
            ros: ros,
            rootObject: viewer.scene
        });
        zoom.startZoom(250, 250);
        zoom.zoom(1.2);
    }

    window.zoomOutMap = function(ros, viewer) {
        var zoom = new ROS2D.ZoomView({
            ros: ros,
            rootObject: viewer.scene
        });
        zoom.startZoom(250, 250);
        zoom.zoom(0.8);
    }


    $("#home").click(function(event) {
        event.preventDefault();
        console.log("home button clicked");
        window.navigation = false;
        window.homing = true;
    });

    $("#navigate").click(function(event) {
        event.preventDefault();
        console.log("navigate button clicked");
        window.navigation = true;
        window.homing = false;
    });

    $('#path').click(function() {
    	path = !path;
    	console.log(path);
        event.preventDefault();
        if (path) {
            pathed();
        } else {
            upathed();
        }
    });
    $("#zoomplus").click(function(event) {
        event.preventDefault();
        zoomInMap(ros, viewer);

    });

    $("#zoomminus").click(function(event) {
        event.preventDefault();
        zoomOutMap(ros, viewer);

    });

    $("#maplive").click(function(event) {
        event.preventDefault();
        console.log("clicked");
        mapchangelive();

    });

    $("#mapgraphical").click(function(event) {
        event.preventDefault();
        console.log("clicked");
        mapchangegraphical();

    });
    $("#stop").click(function(event) {
        event.preventDefault();
        console.log("clicked");
        $.ajax({
            url: '/navigation/stop',
            type: 'POST',
            success: function(response) {
                console.log(response);
            },
            error: function(error) {
                console.log(error);
            }
        })
        upathed();

    });

    $("#shutdown").click(function(event) {
        event.preventDefault();
        $.ajax({
            url: '/shutdown',
            type: 'POST',
            success: function(response) {
                console.log(response);
            },
            error: function(error) {
                console.log(error);
            }
        })

    });

    $("#restart").click(function(event) {
        event.preventDefault();
        $.ajax({
            url: '/restart',
            type: 'POST',
            success: function(response) {
                console.log(response);
            },
            error: function(error) {
                console.log(error);
            }
        })

    });



    $('.menu-btn').click(function() {
        $(this).toggleClass("menu-btn-left");
        $('.box-out').toggleClass('box-in');
    });

    $(".close-navigation").click(function(event) {
        event.stopPropagation();
        // console.log(event.target.previousSibling.data)
        $.ajax({
            url: '/navigation/deletemap',
            type: 'POST',
            data: event.target.previousSibling.data,
            success: function(response) {
                console.log(response);
                location.reload()
            },
            error: function(error) {
                console.log(error);
            }
        })

    });

    var close = document.getElementsByClassName("close-navigation");
    var i;
    for (i = 0; i < close.length; i++) {
        close[i].onclick = function() {
            var div = this.parentElement;
            div.style.display = "none";
        }
    }



});

/* When the user clicks on the button, 
toggle between hiding and showing the dropdown content */
function myFunction() {
    document.getElementById("drop-text").classList.toggle("show");
}

// Close the dropdown if the user clicks outside of it
window.onclick = function(e) {
    if (e.target.matches('.dropbtn')) {
        var myDropdown = document.getElementById("drop-text");
        if (myDropdown.classList.contains('show')) {
            myDropdown.classList.remove('show');
        }
    }
}
