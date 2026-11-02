// SPDX-License-Identifier: Apache-2.0
// Compile the real bootstrap/socket/Linux transport sources. No CUDA is executed.
#include "../src/bootstrap.cc"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <sys/socket.h>
#include <unistd.h>

uint32_t ncclDebugLevelMask = 0;
uint64_t ncclDebugMask = 0;
thread_local int ncclDebugNoWarn = 0;
void ncclDebugLogInternal(ncclDebugLogLevel, unsigned long, const char*, const char*, int, const char*, ...) {}
int64_t ncclLoadParam(const char*, int64_t value, int64_t, int64_t*, int8_t*) { return value; }

static void adopt(ncclSocket* sock, int fd) {
  assert(ncclSocketInit(sock, nullptr, NCCL_SOCKET_MAGIC) == ncclSuccess);
  assert(ncclSocketSetFd(fd, sock) == ncclSuccess);
  // socketpair supplies a preconnected stream; the test bypasses only the hello.
  sock->state = ncclSocketStateReady;
}

int main(int argc, char** argv) {
  assert(argc == 2);
  bool closed = strcmp(argv[1], "closed") == 0;
  int forward[2], backward[2];
  assert(socketpair(AF_UNIX, SOCK_STREAM, 0, forward) == 0);
  assert(socketpair(AF_UNIX, SOCK_STREAM, 0, backward) == 0);
  ncclComm comm[2]{};
  bootstrapState state[2]{};
  adopt(&state[0].ring.socket.send, forward[0]);
  adopt(&state[1].ring.socket.recv, forward[1]);
  adopt(&state[1].ring.socket.send, backward[0]);
  adopt(&state[0].ring.socket.recv, backward[1]);
  ncclSocketAddress peers[2][2];
  ncclResult_t actual[2]{};
  ncclResult_t transportResult = ncclSuccess;
  for (int rank = 0; rank < 2; rank++) {
    comm[rank].rank = state[rank].rank = rank;
    comm[rank].nRanks = state[rank].nranks = 2;
    memset(peers[rank], 0x33 + rank, sizeof(peers[rank]));
  }
  if (closed) {
    assert(ncclSocketClose(&state[1].ring.socket.recv) == ncclSuccess);
    assert(ncclSocketClose(&state[1].ring.socket.send) == ncclSuccess);
    int scratch[2]{};
    transportResult = bootstrapAllGather(&state[0], scratch, sizeof(scratch[0]));
    assert(transportResult == ncclRemoteError);
    actual[0] = ringAllInfo(&comm[0], &state[0], peers[0], nullptr, nullptr, nullptr);
  } else {
    std::thread remote([&]() {
      actual[1] = ringAllInfo(&comm[1], &state[1], peers[1], nullptr, nullptr, nullptr);
    });
    actual[0] = ringAllInfo(&comm[0], &state[0], peers[0], nullptr, nullptr, nullptr);
    remote.join();
  }
  bool outputCorrect = true;
  for (int rank = 0; rank < (closed ? 1 : 2); rank++) {
    for (int peer = 0; peer < 2; peer++) {
      auto* bytes = reinterpret_cast<const unsigned char*>(&peers[rank][peer]);
      for (size_t i = 0; i < sizeof(peers[rank][peer]); i++)
        outputCorrect &= bytes[i] == 0x33 + (closed ? rank : peer);
    }
    assert(ncclSocketClose(&state[rank].ring.socket.recv) == ncclSuccess);
    assert(ncclSocketClose(&state[rank].ring.socket.send) == ncclSuccess);
  }
  ncclResult_t expected = closed ? ncclRemoteError : ncclSuccess;
  bool resultsCorrect = actual[0] == expected && (closed || actual[1] == expected);
  printf("socket=%s expected=%d transport=%d actual=%d,%d output=%s result=%s\n", argv[1], expected,
         transportResult, actual[0], actual[1], outputCorrect ? "PASS" : "FAIL", resultsCorrect ? "PASS" : "FAIL");
  return resultsCorrect && outputCorrect ? 0 : 1;
}
