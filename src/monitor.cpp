#include"monitor.hpp"

Monitor::Monitor(const std::string& host, int port, const std::string& pass) : keystack(), keymap()
{
	constexpr int RESOLUTION_WIDTH = 1360;
	constexpr int RESOLUTION_HEIGHT = 768;

	pwd = pass;

	cl = rfbGetClient(8, 3 ,4);
	cl->format.depth = 24;
	cl->format.bitsPerPixel = 32;
	cl->format.redShift = 16;
	cl->format.greenShift = 8;
	cl->format.blueShift = 0;
	cl->format.redMax = 0xff;
	cl->format.greenMax = 0xff;
	cl->format.blueMax = 0xff;
	cl->appData.compressLevel = 9;
	cl->appData.qualityLevel = 1;
	cl->appData.encodingsString = "tight ultra";
	cl->serverHost = strdup(host.c_str());
	cl->serverPort = port;
	cl->GetPassword = get_password_callback;

	rfbClientSetClientData(cl, nullptr, this);

	pointer_x = pointer_y = 100;
	mouse_r = mouse_l = false;

	if(!rfbInitClient(cl, 0, nullptr))
	{
		throw std::runtime_error("Error thrown on constructing Monitor");
	}
}

char* Monitor::get_password_callback(rfbClient* cl)
{
	Monitor* monitor = static_cast<Monitor*>(rfbClientGetClientData(cl, nullptr));
	return strdup(monitor->password().c_str());
}

bool Monitor::recieve()
{
	int i = WaitForMessage(cl, 500);
	if(i < 0)
	{
		close();
		return false;
	}

	if(i && !HandleRFBServerMessage(cl))
	{
		close();
		return false;
	}

	return true;
}

void Monitor::close()
{
	rfbClientCleanup(cl);
}

void Monitor::control(float x, float y, bool l, bool r)
{
	while(keystack.size() > 0)
	{
		const Key106 k = keystack.top();
		const auto value = keymap[k];
		if(!SendKeyEvent(cl, value, true)) throw std::runtime_error("c");

		keystack.pop();
	}
	
	int mask = 0;
	if(l) mask |= rfbButton1Mask;
	if(r) mask |= rfbButton2Mask;

	pointer_x += x;
	pointer_y += y;

	SendPointerEvent(cl, static_cast<int>(pointer_x), static_cast<int>(pointer_y), mask);
}

void Monitor::push_key(Key106 k)
{
	keystack.push(k);
}
