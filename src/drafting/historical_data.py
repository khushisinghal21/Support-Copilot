"""Comprehensive seed historical resolutions representing authentic @AppleSupport response patterns."""

HISTORICAL_APPLE_RESOLUTIONS: list[dict[str, str]] = [
    # OS & Software Troubleshooting
    {
        "tweet_id": "hist_os_001",
        "customer_text": "My iPhone 11 keeps freezing and won't respond to touch after the update.",
        "agent_reply": "We'd like to help get this sorted out. Have you tried a force restart? Check the exact steps here: apple.co/forcerestart",
        "intent": "OS_SOFTWARE_TROUBLESHOOTING",
    },
    {
        "tweet_id": "hist_os_002",
        "customer_text": "Wi-Fi keeps disconnecting every few minutes on my iPhone X.",
        "agent_reply": "We're here to help. Does this happen on all Wi-Fi networks or just your home network? Try resetting network settings: apple.co/networksettings",
        "intent": "OS_SOFTWARE_TROUBLESHOOTING",
    },
    {
        "tweet_id": "hist_os_003",
        "customer_text": "My iPad screen goes black when opening the Netflix app.",
        "agent_reply": "Let's look into this together. Please check the App Store to ensure all apps are updated to the latest version: apple.co/appupdates",
        "intent": "OS_SOFTWARE_TROUBLESHOOTING",
    },
    {
        "tweet_id": "hist_os_004",
        "customer_text": "My iPhone is stuck in a reboot loop with the Apple logo appearing and disappearing.",
        "agent_reply": "We understand how concerning this is. Connect your iPhone to a computer with iTunes/Finder to attempt an update: apple.co/restoreloop",
        "intent": "OS_SOFTWARE_TROUBLESHOOTING",
    },
    {
        "tweet_id": "hist_os_005",
        "customer_text": "Bluetooth disconnects from my car stereo immediately after pairing.",
        "agent_reply": "We'd like to help with Bluetooth in your car. Forget the stereo in Settings > Bluetooth, restart your iPhone, and pair again: apple.co/bluetoothhelp",
        "intent": "OS_SOFTWARE_TROUBLESHOOTING",
    },
    {
        "tweet_id": "hist_os_006",
        "customer_text": "Cannot update to latest iOS, error says unable to verify update.",
        "agent_reply": "Let's help get your device updated. Ensure you have sufficient free storage and a stable Wi-Fi connection: apple.co/iosupdatehelp",
        "intent": "OS_SOFTWARE_TROUBLESHOOTING",
    },
    {
        "tweet_id": "hist_os_007",
        "customer_text": "Safari closes unexpectedly when loading news websites.",
        "agent_reply": "We can help resolve Safari crashing. Try clearing Safari history and website data in Settings > Safari: apple.co/safaricrash",
        "intent": "OS_SOFTWARE_TROUBLESHOOTING",
    },
    {
        "tweet_id": "hist_os_008",
        "customer_text": "My notifications don't make any sound even with ringer on.",
        "agent_reply": "Let's check your sound settings. Ensure Do Not Disturb is off and check Settings > Sounds & Haptics: apple.co/soundsettings",
        "intent": "OS_SOFTWARE_TROUBLESHOOTING",
    },
    {
        "tweet_id": "hist_os_009",
        "customer_text": "Camera app is lagging and taking 5 seconds to snap a photo.",
        "agent_reply": "We'd be glad to look into camera performance. Close other open apps and restart your device: apple.co/cameratroubleshoot",
        "intent": "OS_SOFTWARE_TROUBLESHOOTING",
    },
    {
        "tweet_id": "hist_os_010",
        "customer_text": "AirDrop fails when trying to send photos to my friend's Mac.",
        "agent_reply": "Let's get AirDrop working. Ensure both Wi-Fi and Bluetooth are turned on and AirDrop is set to Everyone: apple.co/airdropissues",
        "intent": "OS_SOFTWARE_TROUBLESHOOTING",
    },
    {
        "tweet_id": "hist_os_011",
        "customer_text": "Music app stops playing songs after 30 seconds of screen lock.",
        "agent_reply": "We'd like to help with Apple Music. Check Settings > Music and toggle Background App Refresh: apple.co/apptroubleshoot",
        "intent": "OS_SOFTWARE_TROUBLESHOOTING",
    },
    {
        "tweet_id": "hist_os_012",
        "customer_text": "Alarm didn't go off this morning and made me late for work!",
        "agent_reply": "We understand how frustrating that is. Check your alarm sound volume and ensure the ringer switch is set correctly: apple.co/clockhelp",
        "intent": "OS_SOFTWARE_TROUBLESHOOTING",
    },
    {
        "tweet_id": "hist_os_013",
        "customer_text": "Keyboard typing is delayed and drops keystrokes on Messages.",
        "agent_reply": "Let's resolve keyboard lag. Try resetting the keyboard dictionary in Settings > General > Reset: apple.co/keyboardsettings",
        "intent": "OS_SOFTWARE_TROUBLESHOOTING",
    },
    {
        "tweet_id": "hist_os_014",
        "customer_text": "My phone restarts every time I plug in headphones.",
        "agent_reply": "We can help troubleshoot unexpected restarts. Inspect the headphone adapter and update your iOS version: apple.co/forcerestart",
        "intent": "OS_SOFTWARE_TROUBLESHOOTING",
    },
    # Hardware & Battery Issues
    {
        "tweet_id": "hist_hw_001",
        "customer_text": "My battery percentage drops from 80% to 20% within an hour.",
        "agent_reply": "We'd like to check this with you. Please go to Settings > Battery > Battery Health and review maximum capacity: apple.co/batteryhealth",
        "intent": "HARDWARE_AND_BATTERY",
    },
    {
        "tweet_id": "hist_hw_002",
        "customer_text": "The phone won't charge unless I wiggle the charging cable.",
        "agent_reply": "Let's troubleshoot your charging port. Inspect the port gently with a flashlight for any lint or debris: apple.co/charginghelp",
        "intent": "HARDWARE_AND_BATTERY",
    },
    {
        "tweet_id": "hist_hw_003",
        "customer_text": "I dropped my iPhone and the front glass is completely shattered.",
        "agent_reply": "We're sorry to hear about your screen. You can check repair estimates and book an appointment at your nearest Apple Store here: apple.co/repairpricing",
        "intent": "HARDWARE_AND_BATTERY",
    },
    {
        "tweet_id": "hist_hw_004",
        "customer_text": "The bottom speaker sounds crackly and muffled during calls.",
        "agent_reply": "We're happy to help with sound issues. Please check that the speaker mesh is clean and clear of debris: apple.co/cleanspeakers",
        "intent": "HARDWARE_AND_BATTERY",
    },
    {
        "tweet_id": "hist_hw_005",
        "customer_text": "My iPhone is overheating and gets too hot to hold while charging.",
        "agent_reply": "We want to ensure your device stays cool. Check out acceptable operating temperatures and charging guidelines: apple.co/temperaturewarning",
        "intent": "HARDWARE_AND_BATTERY",
    },
    {
        "tweet_id": "hist_hw_006",
        "customer_text": "The volume up button is physically jammed and won't click.",
        "agent_reply": "We can help you arrange physical service for sticky buttons. Check warranty status and service options: apple.co/hardwarehelp",
        "intent": "HARDWARE_AND_BATTERY",
    },
    {
        "tweet_id": "hist_hw_007",
        "customer_text": "My screen has green vertical lines running down the display.",
        "agent_reply": "We can help arrange a display inspection at an Apple Authorized Service Provider: apple.co/screenrepair",
        "intent": "HARDWARE_AND_BATTERY",
    },
    # Account, Billing & iCloud
    {
        "tweet_id": "hist_acc_001",
        "customer_text": "My Apple ID is locked for security reasons and I can't reset password.",
        "agent_reply": "We can help you regain access. Please visit iforgot.apple.com to verify your identity and safely reset your password: apple.co/passwordreset",
        "intent": "ACCOUNT_BILLING_ICLOUD",
    },
    {
        "tweet_id": "hist_acc_002",
        "customer_text": "I was billed $14.99 twice for my Apple Music subscription.",
        "agent_reply": "You can review recent purchases and request a refund directly through our secure portal: reportaproblem.apple.com",
        "intent": "ACCOUNT_BILLING_ICLOUD",
    },
    {
        "tweet_id": "hist_acc_003",
        "customer_text": "My iCloud says storage full but I deleted all my big videos.",
        "agent_reply": "Let's check your storage usage. Go to Settings > [Your Name] > iCloud > Manage Storage to see what is using space: apple.co/manageicloud",
        "intent": "ACCOUNT_BILLING_ICLOUD",
    },
    {
        "tweet_id": "hist_acc_004",
        "customer_text": "Not receiving the 2-step verification code on my trusted number.",
        "agent_reply": "We'd like to assist with two-factor authentication. Follow these alternate verification steps: apple.co/twofactorhelp",
        "intent": "ACCOUNT_BILLING_ICLOUD",
    },
    {
        "tweet_id": "hist_acc_005",
        "customer_text": "How do I cancel my HBO subscription through iTunes?",
        "agent_reply": "Canceling subscriptions is simple. Go to Settings > [Your Name] > Subscriptions and tap Cancel: apple.co/cancelsubscription",
        "intent": "ACCOUNT_BILLING_ICLOUD",
    },
    # How-To & Configuration
    {
        "tweet_id": "hist_howto_001",
        "customer_text": "How do I transfer photos from my iPhone to my Windows laptop?",
        "agent_reply": "We'd be glad to help! Connect your iPhone to your PC with a USB cable and use the Windows Photos app: apple.co/importphotos",
        "intent": "HOW_TO_CONFIGURATION",
    },
    {
        "tweet_id": "hist_howto_002",
        "customer_text": "How do I set up Apple Pay with my debit card on Apple Watch?",
        "agent_reply": "Setting up Apple Pay is simple! Open the Watch app on your paired iPhone, tap Wallet & Apple Pay, then tap Add Card: apple.co/setupapplepay",
        "intent": "HOW_TO_CONFIGURATION",
    },
    {
        "tweet_id": "hist_howto_003",
        "customer_text": "How do I back up my iPhone before updating to iOS 11?",
        "agent_reply": "Backing up is a great habit! Go to Settings > [Your Name] > iCloud > iCloud Backup, and tap Back Up Now: apple.co/backupiphone",
        "intent": "HOW_TO_CONFIGURATION",
    },
    {
        "tweet_id": "hist_howto_004",
        "customer_text": "How do I turn on Night Shift or Dark Mode in settings?",
        "agent_reply": "You can enable Night Shift in Settings > Display & Brightness > Night Shift: apple.co/nightshift",
        "intent": "HOW_TO_CONFIGURATION",
    },
    {
        "tweet_id": "hist_howto_005",
        "customer_text": "Where can I find the serial number and IMEI on my device?",
        "agent_reply": "You can find your device serial number in Settings > General > About: apple.co/findserial",
        "intent": "HOW_TO_CONFIGURATION",
    },
    {
        "tweet_id": "hist_howto_006",
        "customer_text": "Can I connect two pairs of AirPods to one iPhone at the same time?",
        "agent_reply": "Yes! You can share audio with two pairs of AirPods using Audio Sharing in Control Center: apple.co/audiopairing",
        "intent": "HOW_TO_CONFIGURATION",
    },
    # Out of Scope / Ambiguous
    {
        "tweet_id": "hist_amb_001",
        "customer_text": "Can someone at Apple please help me with my phone?",
        "agent_reply": "We're here and ready to help! Please let us know what specific issue you're experiencing with your device so we can assist.",
        "intent": "OUT_OF_SCOPE_AMBIGUOUS",
    },
]
