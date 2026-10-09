# Repository root: forwards to the Track 2A project root (track_2a/ is the project root as defined by the
# Hack Apertus template). `make run` here == `cd track_2a && make run`.
.PHONY: run build smoke selftest
run build smoke selftest:
	$(MAKE) -C track_2a $@
