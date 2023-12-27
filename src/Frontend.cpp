// Frontend.cpp
// ~~~~~~~~~~~~
// Implementations of frontend pro
#include <Frontend.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// Fast file read with mmap
const char *edge::mapSourceFile(const char *fileName, size_t &length) {
  int fd = open(fileName, O_RDONLY);
  if (fd == -1) {
    handleSourceReadError("Error opening file.", ENOENT);
  }
  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    handleSourceReadError("Error with fstat obtaining filesize.", EIO);
  }

  length = sb.st_size;

  const char *addr = static_cast<const char *>(
      mmap(NULL, length, PROT_READ, MAP_PRIVATE, fd, 0u));

  if (addr == MAP_FAILED) {
    handleSourceReadError("Error with mmap.", EIO);
  }
  return addr;
}
