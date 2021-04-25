#ifndef STRING_UTIL_H_
#define STRING_UTIL_H_

#include <vector>
#include <string>
#include <sstream>
#include "third_party/absl/strings/string_view.h"

// String utilities
namespace string_util {

typedef uint32_t char32;
typedef uint32_t uint32;
static constexpr uint32 kUnicodeError = 0xFFFD;

struct string_view_hash {
  // DJB hash function.
  inline size_t operator()(const absl::string_view &sp) const {
    size_t hash = 5381;
    for (size_t i = 0; i < sp.size(); ++i) {
      hash = ((hash << 5) + hash) + sp[i];
    }
    return hash;
  }
};

template <typename Target>
inline bool lexical_cast(absl::string_view arg, Target *result) {
  std::stringstream ss;
  return (ss << arg.data() && ss >> *result);
}

template <>
inline bool lexical_cast(absl::string_view arg, bool *result) {
  const char *kTrue[] = {"1", "t", "true", "y", "yes"};
  const char *kFalse[] = {"0", "f", "false", "n", "no"};
  std::string lower_value = std::string(arg);
  std::transform(lower_value.begin(), lower_value.end(), lower_value.begin(),
                 ::tolower);
  for (size_t i = 0; i < 5; ++i) {
    if (lower_value == kTrue[i]) {
      *result = true;
      return true;
    } else if (lower_value == kFalse[i]) {
      *result = false;
      return true;
    }
  }

  return false;
}

template <>
inline bool lexical_cast(absl::string_view arg, std::string *result) {
  *result = std::string(arg);
  return true;
}

template <typename T>
inline bool DecodePOD(absl::string_view str, T *result) {
  CHECK_NOTNULL(result);
  if (sizeof(*result) != str.size()) {
    return false;
  }
  memcpy(result, str.data(), sizeof(T));
  return true;
}

template <typename T>
inline std::string EncodePOD(const T &value) {
  std::string s;
  s.resize(sizeof(T));
  memcpy(const_cast<char *>(s.data()), &value, sizeof(T));
  return s;
}

template <typename T>
inline std::string IntToHex(T value) {
  std::ostringstream os;
  os << std::hex << std::uppercase << value;
  return os.str();
}

template <typename T>
inline T HexToInt(absl::string_view value) {
  T n;
  std::istringstream is(value.data());
  is >> std::hex >> n;
  return n;
}

template <typename T>
inline size_t Itoa(T val, char *s) {
  char *org = s;

  if (val < 0) {
    *s++ = '-';
    val = -val;
  }
  char *t = s;

  T mod = 0;
  while (val) {
    mod = val % 10;
    *t++ = static_cast<char>(mod) + '0';
    val /= 10;
  }

  if (s == t) {
    *t++ = '0';
  }

  *t = '\0';
  std::reverse(s, t);
  return static_cast<size_t>(t - org);
}

template <typename T>
std::string SimpleItoa(T val) {
  char buf[32];
  Itoa<T>(val, buf);
  return std::string(buf);
}

// Return length of a single UTF-8 source character
inline size_t OneCharLen(const char *src) {
  return "\1\1\1\1\1\1\1\1\1\1\1\1\2\2\3\4"[(*src & 0xFF) >> 4];
}

// Return (x & 0xC0) == 0x80;
// Since trail bytes are always in [0x80, 0xBF], we can optimize:
inline bool IsTrailByte(char x) { return static_cast<signed char>(x) < -0x40; }

inline bool IsValidCodepoint(char32 c) {
  return (static_cast<uint32>(c) < 0xD800) || (c >= 0xE000 && c <= 0x10FFFF);
}

bool IsStructurallyValid(absl::string_view str);

using UnicodeText = std::vector<char32>;

char32 DecodeUTF8(const char *begin, const char *end, size_t *mblen);

inline char32 DecodeUTF8(absl::string_view input, size_t *mblen) {
  return DecodeUTF8(input.data(), input.data() + input.size(), mblen);
}

inline bool IsValidDecodeUTF8(absl::string_view input, size_t *mblen) {
  const char32 c = DecodeUTF8(input, mblen);
  return c != kUnicodeError || *mblen == 3;
}

size_t EncodeUTF8(char32 c, char *output);

std::string UnicodeCharToUTF8(const char32 c);

UnicodeText UTF8ToUnicodeText(absl::string_view utf8);

std::string UnicodeTextToUTF8(const UnicodeText &utext);

}  // namespace string_util

#endif

