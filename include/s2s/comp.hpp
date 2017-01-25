#include <string>
#include "s2s/define.hpp"

#ifndef INCLUDE_GUARD_COMP_HPP
#define INCLUDE_GUARD_COMP_HPP

namespace s2s {

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
        bool operator()(const std::pair<std::string, unsigned int>& first, const std::pair<std::string, unsigned int>& second) {
            if(first.second > second.second){
                return true;
            }
            return false;
        }
    };

};

#endif // INCLUDE_GUARD_COMP_HPP
