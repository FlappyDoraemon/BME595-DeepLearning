LIBOPTS = -shared
CFLAGS = -fPIC -std=gnu99
CC = gcc
libc_convolution.so : c_convolution.c
	$(CC) $< $(LIBOPTS) $(CFLAGS) -o $@
clean:
	rm *.so

