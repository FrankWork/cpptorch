#ifndef UTIL_H_
#define UTIL_H_

#include <string>
#include <sstream>


namespace error{

enum Code{
	OK=0,
	CANCELLED = 1,
  UNKNOWN = 2,
  INVALID_ARGUMENT = 3,
  DEADLINE_EXCEEDED = 4,
  NOT_FOUND = 5,
  ALREADY_EXISTS = 6,
  PERMISSION_DENIED = 7,
  UNAUTHENTICATED = 16,
  RESOURCE_EXHAUSTED = 8,
  FAILED_PRECONDITION = 9,
  ABORTED = 10,
  OUT_OF_RANGE = 11,
  UNIMPLEMENTED = 12,
  INTERNAL = 13,
  UNAVAILABLE = 14,
  DATA_LOSS = 15,
};

}// namespace error

namespace util{


class Status{
	public:
		Status();
		~Status();
		Status(error::Code code, const char*error_message);
		Status(error::Code code, const std::string& error_message);
		Status(const Status& s);
		void operator=(const Status& s);
		void operator==(const Status& s) const;
		void operator!=(const Status& s) const;
		inline bool ok() const{ return rep_ == nullptr;}

		void set_error_message(const char* str);
		const char* error_message() const;
		error::Code code() const;
		std::string ToString() const;

		void IgnoreError();

	private:
		struct Rep;
		std::unique_ptr<Rep> rep_;
};

class StatusBuilder{
	public:
		explicit StatusBuilder(error::Code code):code_(code){}

		template <typename T>
		StatusBuilder& operator<<(const T& value){
			os_ << value;
			return *this;
		}

		operator Status() const { return Status(code_, os_.str());}
	private:
		error::Code code_;
		std::ostringstream os_;
};

};// namespace util

#endif  // UTIL_H_
