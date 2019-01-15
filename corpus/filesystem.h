#ifndef FILESYSTEM_H_
#define FILESYSTEM_H_

#include "util.h"
#include "third_party/absl/strings/string_view.h"

namespace filesystem{

class ReadableFile{
	public:
		ReadableFile(){}
		explicit ReadableFile(absl::string_view filename, bool is_binary=false){}
		virtual ~ReadableFile(){}

		virtual util::Status status() const=0;
		virtual bool ReadLine(std::string *line)=0;
		virtual bool ReadAll(std::string *line=0;
};

class WritableFile{
	public:
		WritableFile(){}
		explicit WritableFile(absl::string_view filename, bool is_binary=false){}
		virtual ~WritableFile(){}

		virtual util::Status status() const=0;
		virtual bool Write(absl::string_view text)=0;
		virtual bool WriteLine(absl::string_view text)=0;
};

std::unique_ptr<ReadableFile> NewReadableFile(absl::string_view filename,
											  bool is_binary=false);

std::unique_ptr<WritableFile> NewWritableFile(absl::string_view filename,
											  bool is_binary=false);
}// namespace filesystem

#endif //FILESYSTEM_H_
