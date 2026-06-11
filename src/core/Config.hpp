#pragma once

// Minimal INI-style case-file parser (no dependencies):
//   # comment, ; comment
//   key = value
//   [section]          -> keys below are looked up as "section.key"
// Lookups are typed, with a default or required (throwing) form.

#include <cstdlib>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>

namespace mm {

class Config {
public:
    static Config load(const std::string& path) {
        std::ifstream in(path);
        if (!in) throw std::runtime_error("cannot open config: " + path);
        Config c;
        std::string line, section;
        int ln = 0;
        while (std::getline(in, line)) {
            ++ln;
            const auto hash = line.find_first_of("#;");
            if (hash != std::string::npos) line.erase(hash);
            const std::string s = trim_(line);
            if (s.empty()) continue;
            if (s.front() == '[') {
                if (s.back() != ']')
                    fail_(path, ln, "malformed section header");
                section = trim_(s.substr(1, s.size() - 2));
                continue;
            }
            const auto eq = s.find('=');
            if (eq == std::string::npos)
                fail_(path, ln, "expected 'key = value'");
            const std::string key = trim_(s.substr(0, eq));
            const std::string val = trim_(s.substr(eq + 1));
            if (key.empty()) fail_(path, ln, "empty key");
            c.kv_[section.empty() ? key : section + "." + key] = val;
        }
        return c;
    }

    bool has(const std::string& key) const { return kv_.count(key) > 0; }

    std::string getString(const std::string& key,
                          const std::string& def) const {
        const auto it = kv_.find(key);
        return it == kv_.end() ? def : it->second;
    }
    std::string requireString(const std::string& key) const {
        const auto it = kv_.find(key);
        if (it == kv_.end())
            throw std::runtime_error("missing required config key: " + key);
        return it->second;
    }

    double getReal(const std::string& key, double def) const {
        const auto it = kv_.find(key);
        return it == kv_.end() ? def : toReal_(key, it->second);
    }

    int getInt(const std::string& key, int def) const {
        const auto it = kv_.find(key);
        if (it == kv_.end()) return def;
        return int(toReal_(key, it->second));
    }

    bool getBool(const std::string& key, bool def) const {
        const auto it = kv_.find(key);
        if (it == kv_.end()) return def;
        const std::string& v = it->second;
        if (v == "true" || v == "1" || v == "on" || v == "yes") return true;
        if (v == "false" || v == "0" || v == "off" || v == "no")
            return false;
        throw std::runtime_error("config key '" + key +
                                 "': not a boolean: " + v);
    }

private:
    static std::string trim_(const std::string& s) {
        const auto a = s.find_first_not_of(" \t\r");
        if (a == std::string::npos) return {};
        const auto b = s.find_last_not_of(" \t\r");
        return s.substr(a, b - a + 1);
    }
    [[noreturn]] static void fail_(const std::string& path, int ln,
                                   const std::string& what) {
        throw std::runtime_error(path + ":" + std::to_string(ln) + ": " +
                                 what);
    }
    static double toReal_(const std::string& key, const std::string& v) {
        char* end = nullptr;
        const double x = std::strtod(v.c_str(), &end);
        if (end == v.c_str() || *end != '\0')
            throw std::runtime_error("config key '" + key +
                                     "': not a number: " + v);
        return x;
    }

    std::map<std::string, std::string> kv_;
};

} // namespace mm
