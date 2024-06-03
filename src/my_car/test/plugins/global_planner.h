#include <ros/ros.h>
#include <costmap_2d/costmap_2d_ros.h>
#include <costmap_2d/costmap_2d.h>
#include <nav_core/base_global_planner.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/transform_broadcaster.h>
#include <angles/angles.h>
#include <base_local_planner/world_model.h>
#include <base_local_planner/costmap_model.h>
#include <nav_msgs/Path.h>

using std::string;

#ifndef GLOBAL_PLANNER_CPP
#define GLOBAL_PLANNER_CPP

namespace global_planner {

class GlobalPlanner : public nav_core::BaseGlobalPlanner {
public:

 GlobalPlanner();
 GlobalPlanner(std::string name, costmap_2d::Costmap2DROS* costmap_ros);

 /** overridden classes from interface nav_core::BaseGlobalPlanner **/
 void initialize(std::string name, costmap_2d::Costmap2DROS* costmap_ros);
 bool makePlan(const geometry_msgs::PoseStamped& start,
               const geometry_msgs::PoseStamped& goal,
               std::vector<geometry_msgs::PoseStamped>& plan
              );
 ros::Publisher plan_pub_;
 
 bool initialized_;
 costmap_2d::Costmap2DROS* costmap_ros_;  //!< costmap ros wrapper
 costmap_2d::Costmap2D* costmap_;  //!< costmap container
 base_local_planner::WorldModel* world_model_;  //!< world model
 };
};
#endif
