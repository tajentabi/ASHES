# ASHES
Autonomous Spectral Hydrogen Emission Scanner (ASHES) Repo for updating production code

# The Plan
Run sdrstack once it had continuous run support, so it periodically integrates and stores data from an RTLSDR to make a dataset that can be analysed with analyze.py after the sweep scan is done

# To-Do
- Support continuous running during a set amount of time to create a set of npz files integrated per another defined duration
- Rotctl support for rotator controls (more data)
- Test system
