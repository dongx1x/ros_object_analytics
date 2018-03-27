/*
 * Copyright (c) 2018 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// System headers
#include <dirent.h>
#include <algorithm>
#include <string>
#include <vector>
// ROS headers
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <object_analytics_msgs/TrackedObjects.h>
#include <object_msgs/ObjectsInBoxes.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
// Non-ROS headers
#include <opencv2/opencv.hpp>
// Local headers
#include "tests/unittest_util.h"

static const std::string kTopicRgb = "rgb";
static int seq = 0;
static ros::Publisher _pub;
static std::vector<std::string> _list;
static std::string _subset;
static object_msgs::ObjectsInBoxesConstPtr dobjs = nullptr;

static std::map<int, cv::Rect2d> trect;
static std::map<int, float> overlaps;
static float overlap = 0.0;
static float overlap_score = 0.0;
static float success_rate_list[21] = {0.0};
static float error_num = 0.0;

using SubscribeImg = message_filters::Subscriber<sensor_msgs::Image>;
using SubscribeTracks =
    message_filters::Subscriber<object_analytics_msgs::TrackedObjects>;
using SubscribeDetect =
    message_filters::Subscriber<object_msgs::ObjectsInBoxes>;
using SyncTracking =
    message_filters::TimeSynchronizer<sensor_msgs::Image,
                                      object_analytics_msgs::TrackedObjects,
                                      object_msgs::ObjectsInBoxes>;

static void getFileList()
{
  dirent* fp = NULL;
  char path[512];
  snprintf(path, sizeof(path), "%s/%s/img", RESOURCE_DIR, _subset.c_str());
  DIR* d = opendir(path);
  while ((fp = readdir(d)) != NULL)
  {
    if (strstr(fp->d_name, ".jpg"))
    {
      _list.push_back(fp->d_name);
    }
  }
  closedir(d);
  std::sort(_list.begin(), _list.end());
}

static void getOverlaps()
{
  char path[512];
  int x = 0, y = 0, w = 0, h = 0, index = 1;
  FILE* fp = NULL;
  snprintf(path, sizeof(path), "%s/%s/groundtruth_rect.txt", RESOURCE_DIR,
           _subset.c_str());
  fp = fopen(path, "r+");
  while (!feof(fp))
  {
    int ret = fscanf(fp, "%d,%d,%d,%d", &x, &y, &w, &h);
    if (ret == 4 && trect.find(index) != trect.end())
    {
      cv::Rect2i ir1(trect[index]);
      cv::Rect2i ir2(cv::Rect2d(x, y, w, h));
      float a1 = ir1.area(), a2 = ir2.area(), a0 = (ir1 & ir2).area();
      float overlap = a0 / (a1 + a2 - a0);
      overlaps.insert(std::pair<int, float>(index, overlap));
    }
    index += 1;
  }
  fclose(fp);
}

static void getAllScores()
{
  int overlap_count = 0;
  int error_count = 0;
  int success_rate_count[21] = {0};
  float overlap_sum = 0.0;
  for (auto i = overlaps.begin(); i != overlaps.end(); i++)
  {
    if (i->second > 0)
    {
      overlap_sum += i->second;
      overlap_count += 1;
    }
    if (i->second < 0.5)
    {
      error_count += 1;
    }
    for (int j = 0; j < 21; j++)
    {
      if (i->second > ((float)j / 20))
      {
        success_rate_count[j] += 1;
      }
    }
  }
  overlap_score = overlap_sum / overlap_count;
  overlap = overlap_score * 100;
  error_num = error_count / (float)overlaps.size();
  for (int j = 0; j < 21; j++)
  {
    success_rate_list[j] = success_rate_count[j] / (float)overlaps.size();
  }
}

static void writeScoresToFile()
{
  char path[512];
  char buf[1024];
  snprintf(path, sizeof(path), "%s/%s/ALL.json", RESOURCE_DIR, _subset.c_str());
  snprintf(buf, sizeof(buf),
           "{\"name\": \"ALL\", \"desc\": \"All attributes\", \"tracker\": "
           "\"MIL\", \"evalType\": \"OPE\", \"seqs\": [\"Human9\"], "
           "\"overlap\": %0.18f, \"error\": %0.18f, \"overlapScores\": "
           "[%0.18f], \"errorNum\": [%0.18f], \"successRateList\": [%0.18f, "
           "%0.18f, %0.18f, %0.18f, %0.18f, %0.18f, %0.18f, %0.18f, %0.18f, "
           "%0.18f, %0.18f, %0.18f, %0.18f, %0.18f, %0.18f, %0.18f, %0.18f, "
           "%0.18f, %0.18f, %0.18f, %0.18f]}",
           overlap, error_num, overlap_score, error_num, success_rate_list[0],
           success_rate_list[1], success_rate_list[2], success_rate_list[3],
           success_rate_list[4], success_rate_list[5], success_rate_list[6],
           success_rate_list[7], success_rate_list[8], success_rate_list[9],
           success_rate_list[10], success_rate_list[11], success_rate_list[12],
           success_rate_list[13], success_rate_list[14], success_rate_list[15],
           success_rate_list[16], success_rate_list[17], success_rate_list[18],
           success_rate_list[19], success_rate_list[20]);
  FILE *fp = fopen(path, "w");
  fwrite(buf, 1, strlen(buf), fp);
  fclose(fp);
}

static void publish()
{
  ros::Rate loop_rate(4);
  char path[512];
  for (auto f : _list) {
    snprintf(path, sizeof(path), "%s/%s/img/%s", RESOURCE_DIR, _subset.c_str(),
             f.c_str());
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    sensor_msgs::ImagePtr msg =
        cv_bridge::CvImage(std_msgs::Header(), "bgr8", img).toImageMsg();
    msg->width = img.cols;
    msg->height = img.rows;
    msg->is_bigendian = false;
    msg->step = img.step;
    msg->header.frame_id = "otb_color_frame";
    msg->header.stamp = ros::Time::now();
    msg->header.seq = seq++;
    _pub.publish(msg);
    ros::spinOnce();
    loop_rate.sleep();
  }
}

static void tracking_cb(
    const sensor_msgs::ImageConstPtr& rgb,
    const object_analytics_msgs::TrackedObjectsConstPtr& tracks,
    const object_msgs::ObjectsInBoxesConstPtr& dobjs)
{
  char buf[256];
  cv::Mat mat = cv_bridge::toCvCopy(rgb, "bgr8")->image;
  if (dobjs)
  {
    for (auto d : dobjs->objects_vector)
    {
      object_msgs::Object dobj = d.object;
      if (strcmp(dobj.object_name.c_str(), "person")) continue;
      sensor_msgs::RegionOfInterest droi = d.roi;
      cv::Rect2d r =
          cv::Rect2d(droi.x_offset, droi.y_offset, droi.width, droi.height);
      std::string n = std::string(dobj.object_name.data());
      cv::Scalar red = cv::Scalar(0, 0, 255);
      cv::rectangle(mat, r, red);
      snprintf(buf, sizeof(buf), "%s [%.0f%%]", n.c_str(),
               dobj.probability * 100);
      cv::putText(mat, buf, cv::Point(r.x, r.y + 48), cv::FONT_HERSHEY_PLAIN,
                  1.0, red);
    }
  }
  for (auto t : tracks->tracked_objects)
  {
    sensor_msgs::RegionOfInterest troi = t.roi;
    cv::Rect2d r =
        cv::Rect2d(troi.x_offset, troi.y_offset, troi.width, troi.height);
    cv::Scalar green = cv::Scalar(0, 255, 0);
    cv::rectangle(mat, r, green);
    snprintf(buf, sizeof(buf), "#%d", t.id);
    cv::putText(mat, buf, cv::Point(r.x, r.y + 32), cv::FONT_HERSHEY_SIMPLEX,
                1.0, green);
  }
  if (tracks->tracked_objects.size() > 0)
  {
    sensor_msgs::RegionOfInterest t0roi = tracks->tracked_objects[0].roi;
    trect.insert(std::pair<int, cv::Rect2d>(
        rgb->header.seq,
        cv::Rect2d(t0roi.x_offset, t0roi.y_offset, t0roi.width, t0roi.height)));
  }
  imshow(_subset, mat);
  cv::waitKey(10);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "otb");
  ros::NodeHandle nh;
  ros::NodeHandle nh_otb("otb");

  nh_otb.param("subset", _subset, std::string("otb/pedistrian1"));
  ROS_INFO("subset %s", _subset.c_str());
  _pub = nh_otb.advertise<sensor_msgs::Image>(kTopicRgb, 10);

  SubscribeImg sub_rgb(nh, "/otb/rgb", 30);
  SubscribeTracks sub_track(nh, "/object_analytics/tracking", 30);
  SubscribeDetect sub_detect(nh, "/object_analytics/detection", 30);
  SyncTracking sync_track(sub_rgb, sub_track, sub_detect, 30);
  sync_track.registerCallback(tracking_cb);

  getFileList();

  publish();

  getOverlaps();
  getAllScores();
  writeScoresToFile();

  return 0;
}
