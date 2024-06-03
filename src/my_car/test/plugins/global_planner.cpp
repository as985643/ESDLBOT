#include <pluginlib/class_list_macros.h>
#include "global_planner.h"

//register this planner as a BaseGlobalPlanner plugin
PLUGINLIB_EXPORT_CLASS(global_planner::GlobalPlanner, nav_core::BaseGlobalPlanner)

using namespace std;

//Default Constructor
namespace global_planner {

  GlobalPlanner::GlobalPlanner (){

  }

  GlobalPlanner::GlobalPlanner(std::string name, costmap_2d::Costmap2DROS* costmap_ros){
    initialize(name, costmap_ros);
  }


  void GlobalPlanner::initialize(std::string name, costmap_2d::Costmap2DROS* costmap_ros){
    if(!initialized_)
    {
      ROS_INFO("Start initialized");
      // get the costmap
      costmap_ros_ = costmap_ros; //initialize the costmap_ros_ attribute to the paramete
      ROS_INFO("1");
      costmap_ = costmap_ros_->getCostmap();  //get the costmap_ from costmap_ros_
      ROS_INFO("2");
      // initialize other planner parameters
      //world_model_ = new base_local_planner::CostmapModel(*costmap_);
      ROS_INFO("3");
      ros::NodeHandle pnh("~" + name);
      ROS_INFO("4");
      // load parameters
      //pnh.param("waypoints_per_meter", waypoints_per_meter_, 20);
    
      plan_pub_ = pnh.advertise<nav_msgs::Path>("global_plan", 1);
      initialized_ = true;
      ROS_INFO("Planner has been initialized");
    }
    else
    {
      ROS_WARN("This planner has already been initialized");
    }

  }
  
  float path[][2] = {{3.25, 3.10}, {3.25, 1.60}, {4.35, 1.60}, {4.35, 3.10}, {4.45, 3.10}, {4.45, 1.60}, 
                     {5.55, 1.60}, {5.55, 3.10}, {5.65, 3.10}, {5.65, 1.60}, {6.75, 1.60}, {6.75, 3.10}};
  
  bool GlobalPlanner::makePlan(const geometry_msgs::PoseStamped& start, const geometry_msgs::PoseStamped& goal,  std::vector<geometry_msgs::PoseStamped>& plan ){

    plan.push_back(start);
    nav_msgs::Path path_;
    //path.clear();
    for (int i=0; i<12; i++){
      geometry_msgs::PoseStamped new_goal = goal;
      tf::Quaternion goal_quat = tf::createQuaternionFromYaw(1.54);

      new_goal.pose.position.x = path[i][0];
      new_goal.pose.position.y = path[i][1];
      //new_goal.pose.position.x = -2.5+(0.05*i);
      //new_goal.pose.position.y = -3.5+(0.05*i);

      new_goal.pose.orientation.x = goal_quat.x();
      new_goal.pose.orientation.y = goal_quat.y();
      new_goal.pose.orientation.z = goal_quat.z();
      new_goal.pose.orientation.w = goal_quat.w();

      plan.push_back(new_goal);
    }
    plan.push_back(goal);
    //path_.poses.insert(path_.poses.begin(), plan.begin(), plan.end());
    //plan_pub_.publish(path_);
    return true;
  }
};
