import commands
import re
import sys


# Confirm this is running on Linux/Mac OS platform
available_platform = ['darwin', 'linux']
if sum([x in sys.platform for x in available_platform]) == 0:
    print("Operating system not supported. Only %s supported" % ",".join(available_platform))
    sys.exit()
else:
    print("Current Operating System: %s" % sys.platform)


# Check if nmap is installed
check_nmap_installed = commands.getstatusoutput("nmap -version")

if check_nmap_installed[0] != 0:
    print("'nmap' is not installed or not configured properly on your machine.")
    print("Please visit https://nmap.org/ for more information.")
    # sys.exit()

else:
    nmap_version_installed = re.search("version ([0-9.]*)", check_nmap_installed[1])
    print("nmap %s is installed on your machine." % nmap_version_installed.group())



# Get the current Broadcast
check_broadcast = commands.getstatusoutput("ifconfig | grep cast")
broadcasts = re.findall('[0-9]{1,3}[.][0-9]{1,3}[.][0-9]{1,3}[.]255', check_broadcast[1])
print "%d broadcast(s) found: %s" % (len(broadcasts), ", ".join(broadcasts))


