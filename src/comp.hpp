#include <string>
#include "define.hpp"

#ifndef INCLUDE_GUARD_COMP_HPP
#define INCLUDE_GUARD_COMP_HPP

// Sort in descending order of length
struct CompareString {
  bool operator()(const ParaSent& first, const ParaSent& second) {
    if(
       (first.first.size() > second.first.size()) ||
       (first.first.size() == second.first.size() && first.second.size() > second.second.size())
    ){
      return true;
    }
    return false;
  }
  bool operator()(const Sent& first, const Sent& second) {
    if(first.size() > second.size()){
      return true;
    }
    return false;
  }
  bool operator()(const pair<string, unsigned int>& first, const pair<string, unsigned int>& second) {
    if(first.second > second.second){
      return true;
    }
    return false;
  }
};

#endif // INCLUDE_GUARD_COMP_HPP
