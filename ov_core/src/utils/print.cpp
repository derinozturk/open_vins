/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

/*
 * Extended by tiny-vio project to add structured (JSON) logging support.
 */

#include "print.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <regex>
#include <sstream>

using namespace ov_core;

// Static variable definitions
Printer::PrintLevel Printer::current_print_level = PrintLevel::INFO;
Printer::OutputFormat Printer::current_output_format = OutputFormat::TEXT;
bool Printer::output_format_initialized = false;

void Printer::initOutputFormatFromEnv() {
  if (output_format_initialized) {
    return;
  }
  output_format_initialized = true;

  const char *env_format = std::getenv("OPENVINS_LOG_FORMAT");
  if (env_format != nullptr) {
    std::string format_str(env_format);
    // Convert to lowercase for comparison
    for (char &c : format_str) {
      c = std::tolower(c);
    }
    if (format_str == "json") {
      current_output_format = OutputFormat::JSON;
    } else if (format_str == "text") {
      current_output_format = OutputFormat::TEXT;
    }
    // Silently ignore invalid values, keep default
  }
}

void Printer::setPrintLevel(const std::string &level) {
  if (level == "ALL")
    setPrintLevel(PrintLevel::ALL);
  else if (level == "DEBUG")
    setPrintLevel(PrintLevel::DEBUG);
  else if (level == "INFO")
    setPrintLevel(PrintLevel::INFO);
  else if (level == "WARNING")
    setPrintLevel(PrintLevel::WARNING);
  else if (level == "ERROR")
    setPrintLevel(PrintLevel::ERROR);
  else if (level == "SILENT")
    setPrintLevel(PrintLevel::SILENT);
  else {
    std::cout << "Invalid print level requested: " << level << std::endl;
    std::cout << "Valid levels are: ALL, DEBUG, INFO, WARNING, ERROR, SILENT" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

void Printer::setPrintLevel(PrintLevel level) {
  Printer::current_print_level = level;

  // Initialize output format from env if not already done
  initOutputFormatFromEnv();

  // Only print level change message in TEXT mode
  if (current_output_format == OutputFormat::TEXT) {
    std::cout << "Setting printing level to: ";
    switch (current_print_level) {
    case PrintLevel::ALL:
      std::cout << "ALL";
      break;
    case PrintLevel::DEBUG:
      std::cout << "DEBUG";
      break;
    case PrintLevel::INFO:
      std::cout << "INFO";
      break;
    case PrintLevel::WARNING:
      std::cout << "WARNING";
      break;
    case PrintLevel::ERROR:
      std::cout << "ERROR";
      break;
    case PrintLevel::SILENT:
      std::cout << "SILENT";
      break;
    default:
      std::cout << std::endl;
      std::cout << "Invalid print level requested: " << level << std::endl;
      std::cout << "Valid levels are: ALL, DEBUG, INFO, WARNING, ERROR, SILENT" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    std::cout << std::endl;
  }
}

void Printer::setOutputFormat(const std::string &format) {
  if (format == "json" || format == "JSON") {
    setOutputFormat(OutputFormat::JSON);
  } else if (format == "text" || format == "TEXT") {
    setOutputFormat(OutputFormat::TEXT);
  } else {
    std::cout << "Invalid output format requested: " << format << std::endl;
    std::cout << "Valid formats are: text, json" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

void Printer::setOutputFormat(OutputFormat format) {
  Printer::current_output_format = format;
  Printer::output_format_initialized = true;
}

namespace {

// Get current timestamp as Unix time (seconds with microsecond precision)
double getCurrentTimestamp() {
  auto now = std::chrono::system_clock::now();
  auto duration = now.time_since_epoch();
  auto micros = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
  return static_cast<double>(micros) / 1000000.0;
}

// Convert PrintLevel to string
const char *levelToString(Printer::PrintLevel level) {
  switch (level) {
  case Printer::PrintLevel::ALL:
    return "ALL";
  case Printer::PrintLevel::DEBUG:
    return "DEBUG";
  case Printer::PrintLevel::INFO:
    return "INFO";
  case Printer::PrintLevel::WARNING:
    return "WARNING";
  case Printer::PrintLevel::ERROR:
    return "ERROR";
  case Printer::PrintLevel::SILENT:
    return "SILENT";
  default:
    return "UNKNOWN";
  }
}

// Strip ANSI color codes from a string
std::string stripAnsiCodes(const std::string &input) {
  // ANSI escape codes have the form: ESC[...m (where ESC is \033 or \x1b)
  // This regex matches: \033[ followed by any number of digits and semicolons, followed by m
  static const std::regex ansi_regex("\033\\[[0-9;]*m");
  return std::regex_replace(input, ansi_regex, "");
}

// Escape string for JSON output
std::string escapeJsonString(const std::string &input) {
  std::ostringstream ss;
  for (char c : input) {
    switch (c) {
    case '"':
      ss << "\\\"";
      break;
    case '\\':
      ss << "\\\\";
      break;
    case '\n':
      ss << "\\n";
      break;
    case '\r':
      ss << "\\r";
      break;
    case '\t':
      ss << "\\t";
      break;
    case '\b':
      ss << "\\b";
      break;
    case '\f':
      ss << "\\f";
      break;
    default:
      if (static_cast<unsigned char>(c) < 0x20) {
        // Control characters: output as \u00XX
        char buf[8];
        std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
        ss << buf;
      } else {
        ss << c;
      }
      break;
    }
  }
  return ss.str();
}

// Extract basename from a file path
std::string getBasename(const std::string &path) {
  size_t pos = path.find_last_of("/\\");
  if (pos == std::string::npos) {
    return path;
  }
  return path.substr(pos + 1);
}

// Derive a tag from the source file path
// e.g., "ov_msckf/src/..." -> "msckf", "ov_core/src/..." -> "core"
const char *deriveTag(const std::string &path) {
  if (path.find("ov_msckf") != std::string::npos) {
    return "msckf";
  } else if (path.find("ov_core") != std::string::npos) {
    return "core";
  } else if (path.find("ov_eval") != std::string::npos) {
    return "eval";
  } else if (path.find("ov_init") != std::string::npos) {
    return "init";
  }
  return nullptr;
}

} // anonymous namespace

void Printer::debugPrint(PrintLevel level, const char location[], const char line[], const char *format, ...) {
  // Initialize output format from env if not already done
  initOutputFormatFromEnv();

  // Only print for the current debug level
  if (static_cast<int>(level) < static_cast<int>(Printer::current_print_level)) {
    return;
  }

  // Format the message
  char buffer[4096];
  va_list args;
  va_start(args, format);
  int len = std::vsnprintf(buffer, sizeof(buffer), format, args);
  va_end(args);

  // Ensure null termination
  if (len >= static_cast<int>(sizeof(buffer))) {
    buffer[sizeof(buffer) - 1] = '\0';
  }

  std::string message(buffer);

  if (current_output_format == OutputFormat::JSON) {
    // JSON Lines output
    double timestamp = getCurrentTimestamp();
    std::string basename = getBasename(location);
    const char *tag = deriveTag(location);

    // Strip ANSI codes and escape for JSON
    std::string clean_message = stripAnsiCodes(message);
    std::string escaped_message = escapeJsonString(clean_message);

    // Build JSON object
    std::ostringstream json;
    json << "{\"ts\":" << std::fixed << std::setprecision(6) << timestamp;
    json << ",\"level\":\"" << levelToString(level) << "\"";
    json << ",\"file\":\"" << escapeJsonString(basename) << "\"";
    json << ",\"line\":" << line;
    if (tag != nullptr) {
      json << ",\"tag\":\"" << tag << "\"";
    }
    json << ",\"msg\":\"" << escaped_message << "\"}";

    std::cout << json.str() << std::endl;
  } else {
    // Original TEXT output behavior
    // Print the location info first for our debug output
    // Truncate the filename to the max size for the filepath
    if (static_cast<int>(Printer::current_print_level) <= static_cast<int>(Printer::PrintLevel::DEBUG)) {
      std::string path(location);
      std::string base_filename = path.substr(path.find_last_of("/\\") + 1);
      if (base_filename.size() > MAX_FILE_PATH_LEGTH) {
        printf("%s", base_filename.substr(base_filename.size() - MAX_FILE_PATH_LEGTH, base_filename.size()).c_str());
      } else {
        printf("%s", base_filename.c_str());
      }
      printf(":%s ", line);
    }

    // Print the message
    printf("%s", message.c_str());
  }
}
