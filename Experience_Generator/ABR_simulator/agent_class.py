import numpy as np
import random
import itertools
import math
import copy

M_IN_K = 1000
CHUNK_TIL_VIDEO_END_CAP = 30  # nr of chunks of video the same of env.total_num_chunks
VIDEO_BIT_RATE = [0.235, 0.375, 0.560, 0.750, 1.050, 1.750, 2.350, 3, 4.3, 5.8, 8.1, 11.6, 16.8]  # Mbps
DEFAULT_QUALITY = 0
RANDOM_SEED = 42


class Agent(object):
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        self.last_action = None
        self.last_obs = None
        self.last_state = None
        self.last_info = None
        self.DEFAULT_QUALITY = DEFAULT_QUALITY  # default video quality without agent

    # assign obs to self.last_obs
    def take_obs_info(self,obs,info):
        self.last_obs=copy.copy(obs)
        self.last_info=copy.copy(info)

    # take last obs and transform it to state
    def obs_to_state(self):
        state = self.last_obs
        self.last_state = state

    # take self.state and output action
    def do_action(self):
        pass

class Random_agent(Agent):
    def __init__(self,state_space, action_space):
        Agent.__init__(self, state_space, action_space)
        self.first_action = True

    def do_action(self):
        if not self.first_action:
            action = random.randint(0, 5)
            return action
        else:
            self.first_action=False
            return self.DEFAULT_QUALITY

    def reset_state(self):
        self.last_obs = None
        self.first_action = True

class MPC_agent(Agent):
    MPC_FUTURE_CHUNK_COUNT = 3
    PREDICTION_WINDOW = 3
    BUFFER_NORM_FACTOR = 10  # sec
    S_INFO = 5  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
    S_LEN = 8  # take how many frames in the past
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)

    def __init__(self,state_space, action_space):
        Agent.__init__(self, state_space, action_space)
        self.CHUNK_COMBO_OPTIONS = []
        self.first_action = True
        # make chunk combination options
        for combo in itertools.product(range(13), repeat=self.PREDICTION_WINDOW):
            self.CHUNK_COMBO_OPTIONS.append(combo)
        self.video_size = np.load('./park/envs/abr_sim/videos/video_sizes_ToS.npy')

    def obs_to_state(self):
        # retrieve previous state
        if self.last_state is None:
            state = np.zeros((self.S_INFO, self.S_LEN))
        else:
            state = np.array(self.last_state, copy=True)

        # dequeue history record
        state = np.roll(state, -1, axis=1)


        # this should be S_INFO number of terms
        try:
            state[0, -1] = VIDEO_BIT_RATE[int(self.last_obs[4])] / float(np.max(VIDEO_BIT_RATE))
            state[1, -1] = self.last_obs[2] / self.BUFFER_NORM_FACTOR
            state[2, -1] = self.last_info['stall_time'] / M_IN_K
            state[3, -1] = self.last_obs[0] / M_IN_K / M_IN_K # kilo byte / ms
            state[4, -1] = np.minimum(self.last_obs[3], CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
        except ZeroDivisionError:
            # this should occur VERY rarely (1 out of 3000), should be a dash issue
            # in this case we ignore the observation and roll back to an eariler one
            if self.last_state is None:
                state = np.zeros((self.S_INFO, self.S_LEN))
            else:
                state = np.array(self.last_state, copy=True)

        self.last_state=state

    def do_action(self):
        if not self.first_action:
            # pick bitrate according to MPC
            # first get harmonic mean of last 5 bandwidths
            past_bandwidths = self.last_state[3, -5:]
            while past_bandwidths[0] == 0.0:
                past_bandwidths = past_bandwidths[1:]  # trick to avoid 0 througput past info (for initial states)
            # if ( len(state) < 5 ):
            #    past_bandwidths = state[3,-len(state):]
            # else:
            #    past_bandwidths = state[3,-5:]
            bandwidth_sum = 0
            for past_val in past_bandwidths:
                bandwidth_sum += (1 / float(past_val))
            future_bandwidth = 1.0 / (bandwidth_sum / len(past_bandwidths))

            # future chunks length (try 4 if that many remaining)
            last_index = int(CHUNK_TIL_VIDEO_END_CAP-self.last_obs[3]-1) #NB in this case is not the chunk nr but the chunk index, so chunk nr 1 is indexed with 0
            future_chunk_length = self.MPC_FUTURE_CHUNK_COUNT
            if (self.last_obs[3] < 3):
                future_chunk_length = int(self.last_obs[3])

            # all possible combinations of 3 chunk bitrates (13^3 options)
            # iterate over list and for each, compute reward and store max reward combination
            max_reward = -100000000
            best_combo = ()
            start_buffer = float(self.last_obs[2])
            # start = time.time()
            for full_combo in self.CHUNK_COMBO_OPTIONS:
                combo = full_combo[0:future_chunk_length]
                # calculate total rebuffer time for this combination (start with start_buffer and subtract
                # each download time and add 2 seconds in that order)
                curr_rebuffer_time = 0
                curr_buffer = start_buffer
                bitrate_sum = 0
                smoothness_diffs = 0
                last_quality = int(self.last_obs[4])
                for position in range(0, len(combo)):
                    chunk_quality = combo[position]
                    index = last_index + position + 1  # e.g., if last chunk is 3, then first iter is 3+0+1=4
                    #print(last_index)
                    #print(position)

                    download_time = (self.video_size[chunk_quality,index] / 1000000.) / future_bandwidth  # this is MB/MB/s --> seconds NB KByte/ms = MByte/s
                    if (curr_buffer < download_time):
                        curr_rebuffer_time += (download_time - curr_buffer)
                        curr_buffer = 0
                    else:
                        curr_buffer -= download_time
                    curr_buffer += 4

                    # linear reward
                    bitrate_sum += VIDEO_BIT_RATE[chunk_quality]
                    smoothness_diffs += abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality])

                    # log reward
                    # log_bit_rate = np.log(VIDEO_BIT_RATE[chunk_quality] / float(VIDEO_BIT_RATE[0]))
                    # log_last_bit_rate = np.log(VIDEO_BIT_RATE[last_quality] / float(VIDEO_BIT_RATE[0]))
                    # bitrate_sum += log_bit_rate
                    # smoothness_diffs += abs(log_bit_rate - log_last_bit_rate)

                    # hd reward
                    #bitrate_sum += self.BITRATE_REWARD[chunk_quality]
                    #smoothness_diffs += abs(self.BITRATE_REWARD[chunk_quality] - self.BITRATE_REWARD[last_quality])

                    last_quality = chunk_quality
                # compute reward for this combination (one reward per 5-chunk combo)
                # bitrates are in Mbits/s, rebuffer in seconds, and smoothness_diffs in Mbits/s

                # linear reward
                reward = (bitrate_sum) - (4.3*curr_rebuffer_time) - (smoothness_diffs)

                # log reward
                # reward = (bitrate_sum) - (4.3*curr_rebuffer_time) - (smoothness_diffs)

                # hd reward
                #reward = bitrate_sum - (8 * curr_rebuffer_time) - (smoothness_diffs)

                if (reward > max_reward):
                    max_reward = reward
                    best_combo = combo
            # send data to html side (first chunk of best combo)
            bit_rate = 0  # no combo had reward better than -1000000 (ERROR) so send 0
            if (best_combo != ()):  # some combo was good
                bit_rate = str(best_combo[0])

            return int(bit_rate)
        else:
            self.first_action=False
            return self.DEFAULT_QUALITY

    def reset_state(self):
        self.last_action = None
        self.last_obs = None
        self.last_state = None
        self.last_info = None
        self.first_action = True

class Robust_MPC_agent(Agent):
    MPC_FUTURE_CHUNK_COUNT = 5
    CHUNK_TIL_VIDEO_END_CAP = 49  # nr of chunks of video
    BUFFER_NORM_FACTOR = 10  # sec
    S_INFO = 5  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
    S_LEN = 8  # take how many frames in the past
    RANDOM_SEED = 42

    np.random.seed(RANDOM_SEED)

    def __init__(self, state_space, action_space):
        Agent.__init__(self, state_space, action_space)
        self.CHUNK_COMBO_OPTIONS = []
        self.past_errors = []
        self.past_bandwidth_ests = []
        self.first_action = True
        # make chunk combination options
        for combo in itertools.product([0, 1, 2, 3, 4, 5], repeat=5):
            self.CHUNK_COMBO_OPTIONS.append(combo)
        self.video_size = np.load('./park/envs/abr_sim/videos/video_sizes.npy')

    def obs_to_state(self):
        # retrieve previous state
        if self.last_state is None:
            state = np.zeros((self.S_INFO, self.S_LEN))
        else:
            state = np.array(self.last_state, copy=True)

        # dequeue history record
        state = np.roll(state, -1, axis=1)

        # this should be S_INFO number of terms
        try:
            state[0, -1] = VIDEO_BIT_RATE[int(self.last_obs[4])] / float(np.max(VIDEO_BIT_RATE))
            state[1, -1] = self.last_obs[2] / self.BUFFER_NORM_FACTOR
            state[2, -1] = self.last_info['stall_time'] / M_IN_K
            state[3, -1] = self.last_obs[0] / M_IN_K / M_IN_K  # kilo byte / ms
            state[4, -1] = np.minimum(self.last_obs[3], CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
            curr_error = 0  # defualt assumes that this is the first request so error is 0 since we have never predicted bandwidth
            if (len(self.past_bandwidth_ests) > 0):
                curr_error = abs(self.past_bandwidth_ests[-1] - state[3, -1]) / float(state[3, -1])
                #print(state[3,-1])
            self.past_errors.append(curr_error)
        except ZeroDivisionError:
            # this should occur VERY rarely (1 out of 3000), should be a dash issue
            # in this case we ignore the observation and roll back to an eariler one
            self.past_errors.append(0)
            if self.last_state is None:
                state = np.zeros((self.S_INFO, self.S_LEN))
            else:
                state = np.array(self.last_state, copy=True)

        self.last_state = state

    def do_action(self):
        if not self.first_action:
            # pick bitrate according to MPC
            # first get harmonic mean of last 5 bandwidths
            past_bandwidths = self.last_state[3, -5:]
            while past_bandwidths[0] == 0.0:
                past_bandwidths = past_bandwidths[1:]
            # if ( len(state) < 5 ):
            #    past_bandwidths = state[3,-len(state):]
            # else:
            #    past_bandwidths = state[3,-5:]
            bandwidth_sum = 0
            for past_val in past_bandwidths:
                bandwidth_sum += (1 / float(past_val))
            harmonic_bandwidth = 1.0 / (bandwidth_sum / len(past_bandwidths))

            # future bandwidth prediction
            # divide by 1 + max of last 5 (or up to 5) errors
            max_error = 0
            error_pos = -5
            if (len(self.past_errors) < 5):
                error_pos = -len(self.past_errors)
            max_error = float(max(self.past_errors[error_pos:]))
            future_bandwidth = harmonic_bandwidth / (1 + max_error)
            self.past_bandwidth_ests.append(harmonic_bandwidth)

            # future chunks length (try 4 if that many remaining)
            last_index = int(self.CHUNK_TIL_VIDEO_END_CAP - self.last_obs[3]-1)
            future_chunk_length = self.MPC_FUTURE_CHUNK_COUNT
            if (self.last_obs[3] < 5):
                future_chunk_length = int(self.last_obs[3])

            # all possible combinations of 5 chunk bitrates (9^5 options)
            # iterate over list and for each, compute reward and store max reward combination
            max_reward = -100000000
            best_combo = ()
            start_buffer = float(self.last_obs[2])
            # start = time.time()
            for full_combo in self.CHUNK_COMBO_OPTIONS:
                combo = full_combo[0:future_chunk_length]
                # calculate total rebuffer time for this combination (start with start_buffer and subtract
                # each download time and add 2 seconds in that order)
                curr_rebuffer_time = 0
                curr_buffer = start_buffer
                bitrate_sum = 0
                smoothness_diffs = 0
                last_quality = int(self.last_obs[4])
                for position in range(0, len(combo)):
                    chunk_quality = combo[position]
                    index = last_index + position + 1  # e.g., if last chunk is 3, then first iter is 3+0+1=4
                    download_time = (self.video_size[chunk_quality,index] / 1000000.) / future_bandwidth  # this is MB/MB/s --> seconds
                    if (curr_buffer < download_time):
                        curr_rebuffer_time += (download_time - curr_buffer)
                        curr_buffer = 0
                    else:
                        curr_buffer -= download_time
                    curr_buffer += 4

                    # linear reward
                    bitrate_sum += VIDEO_BIT_RATE[chunk_quality]
                    smoothness_diffs += abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality])

                    # log reward
                    # log_bit_rate = np.log(VIDEO_BIT_RATE[chunk_quality] / float(VIDEO_BIT_RATE[0]))
                    # log_last_bit_rate = np.log(VIDEO_BIT_RATE[last_quality] / float(VIDEO_BIT_RATE[0]))
                    # bitrate_sum += log_bit_rate
                    # smoothness_diffs += abs(log_bit_rate - log_last_bit_rate)

                    # hd reward
                    #bitrate_sum += self.BITRATE_REWARD[chunk_quality]
                    #smoothness_diffs += abs(self.BITRATE_REWARD[chunk_quality] - self.BITRATE_REWARD[last_quality])

                    last_quality = chunk_quality
                # compute reward for this combination (one reward per 5-chunk combo)
                # bitrates are in Mbits/s, rebuffer in seconds, and smoothness_diffs in Mbits/s

                # linear reward
                reward = (bitrate_sum) - (4.3 * curr_rebuffer_time) - (smoothness_diffs)

                # log reward
                # reward = (bitrate_sum) - (4.3*curr_rebuffer_time) - (smoothness_diffs)

                # hd reward
                # reward = bitrate_sum - (8 * curr_rebuffer_time) - (smoothness_diffs)

                if (reward > max_reward):
                    max_reward = reward
                    best_combo = combo
            # send data to html side (first chunk of best combo)
            bit_rate = 0  # no combo had reward better than -1000000 (ERROR) so send 0
            if (best_combo != ()):  # some combo was good
                bit_rate = str(best_combo[0])

            return int(bit_rate)
        else:
            self.first_action = False
            return self.DEFAULT_QUALITY

    def reset_state(self):
        self.last_action = None
        self.last_obs = None
        self.last_state = None
        self.last_info = None
        self.first_action = True
        self.past_errors = []
        self.past_bandwidth_ests = []

class Bola_agent(Agent):
    # Bola original uses kbps
    VIDEO_BIT_RATE_BOLA = [i * 1000 for i in VIDEO_BIT_RATE]
    segment_length_t = 4000 # in ms


    def __init__(self, state_space,action_space):
        Agent.__init__(self, state_space, action_space)
        self.first_action = True
        self.mem_past_throughput = []
        utility_offset = -math.log(self.VIDEO_BIT_RATE_BOLA[0]) # so utilities[0] = 0
        self.utilities = [math.log(b) + utility_offset for b in self.VIDEO_BIT_RATE_BOLA]

        # PARAMETERS
        self.gp = 5
        self.buffer_size = 25 * 1000 #in sec pero poi lo converte in ms, it is the maximum allowed buffer
        self.abr_osc = True
        self.abr_basic = True
        self.Vp = (self.buffer_size - self.segment_length_t) / (self.utilities[-1] + self.gp)
        self.last_quality = 0
        self.last_seek_index = 0 # TODO
        # verbose = True
        # if verbose:
        #     for q in range(len(self.VIDEO_BIT_RATE)):
        #         b = self.VIDEO_BIT_RATE[q]
        #         u = self.utilities[q]
        #         l = self.Vp * (self.gp + u)
        #         if q == 0:
        #             print('%d %d' % (q, l))
        #         else:
        #             qq = q - 1
        #             bb = self.VIDEO_BIT_RATE[qq]
        #             uu = self.utilities[qq]
        #             ll = self.Vp * (self.gp + (b * uu - bb * u) / (b - bb))
        #             print('%d %d    <- %d %d' % (q, l, qq, ll))

    def quality_from_throughput(self, tput): #should be in kbit/s but mine is in Byte/sec
        p = self.segment_length_t #chunk length in ms

        quality = 0
        while (quality + 1 < len(self.VIDEO_BIT_RATE_BOLA) and p * self.VIDEO_BIT_RATE_BOLA[quality + 1] / tput <= p): #mi sta dicendo che se per scaricare
            # un chunk ci metto meno che renderizzarlo allora alzo la qualità finche questo non è piu vero, in tutto questo c era anche un parametro di delay che
            # in park non mi viene restituito poiche le traccie non hanno la latenza o delay, non è chiaro#
            quality += 1
        return quality

    def quality_from_buffer(self):
        level = self.last_state[2]
        quality = 0
        score = None
        for q in range(len(self.VIDEO_BIT_RATE_BOLA)):
            s = ((self.Vp * (self.utilities[q] + self.gp) - level) / self.VIDEO_BIT_RATE_BOLA[q])
            if score == None or s > score:
                quality = q
                score = s
        return quality

    def estimated_throughput(self):
        past_bandwidths = self.mem_past_throughput[-5:]
        while past_bandwidths[0] == 0.0:
            past_bandwidths = past_bandwidths[1:]  # trick to avoid 0 througput past info (for initial states)
        # if ( len(state) < 5 ):
        #    past_bandwidths = state[3,-len(state):]
        # else:
        #    past_bandwidths = state[3,-5:]
        bandwidth_sum = 0
        for past_val in past_bandwidths:
            bandwidth_sum += (1 / float(past_val))
        future_bandwidth = 1.0 / (bandwidth_sum / len(past_bandwidths))
        return future_bandwidth


    # it is the basic fucntion for do an action
    def get_quality_delay(self):
        if not self.abr_basic:
            segment_index = CHUNK_TIL_VIDEO_END_CAP-self.last_state[3]
            t = min(segment_index - self.last_seek_index, CHUNK_TIL_VIDEO_END_CAP - segment_index)
            t = max(t / 2, 3)
            t = t * self.segment_length_t
            buffer_size = min(self.buffer_size, t)
            self.Vp = (buffer_size - self.segment_length_t) / (self.utilities[-1] + self.gp)

        quality = self.quality_from_buffer()
        delay = 0

        if quality > self.last_quality:
            thru_est = self.estimated_throughput() # in kbit/s
            quality_t = self.quality_from_throughput(thru_est) #from Bytes/s to kbit/s
            if quality <= quality_t:
                delay = 0
            elif self.last_quality > quality_t:
                quality = self.last_quality
                delay = 0
            else:
                if not self.abr_osc:
                    quality = quality_t + 1
                    delay = 0
                else:
                    quality = quality_t
                    # now need to calculate delay
                    b = self.VIDEO_BIT_RATE_BOLA[quality]
                    u = self.utilities[quality]
                    #bb = manifest.bitrates[quality + 1]
                    #uu = self.utilities[quality + 1]
                    #l = self.Vp * (self.gp + (bb * u - b * uu) / (bb - b))
                    l = self.Vp * (self.gp + u) ##########
                    delay = max(0, self.last_state[2] - l)
                    if quality == len(self.VIDEO_BIT_RATE_BOLA) - 1:
                        delay = 0
                    # delay = 0 ###########

        self.last_quality = quality
        return quality#, delay)

    def do_action(self):
        if not self.first_action:
            return self.get_quality_delay()
        else:
            self.first_action=False
            return self.DEFAULT_QUALITY

    def reset_state(self):
        self.last_state = None
        self.last_obs = None
        self.first_action = True
        self.last_quality = 0
        self.mem_past_throughput = []

    def obs_to_state(self):
        self.last_state=self.last_obs
        self.last_state[2] =self.last_state[2]*1000  # buffer in millissec
        self.mem_past_throughput.append(self.last_state[0] * 0.008) # in kbits

class ThroughputRule_agent(Agent):
    # Bola original uses kbps
    VIDEO_BIT_RATE_T = [i * 1000 for i in VIDEO_BIT_RATE]
    segment_length_t = 4000  # in ms


    def __init__(self, state_space,action_space):
        Agent.__init__(self, state_space, action_space)
        self.safety_factor = 0.9
        self.low_buffer_safety_factor = 0.5
        self.low_buffer_safety_factor_init = 0.9
        self.abandon_multiplier = 1.8
        self.abandon_grace_time = 500
        self.ibr_safety = self.low_buffer_safety_factor_init
        self.no_ibr = True
        self.mem_past_throughput = []
        self.first_action=True

    def estimated_throughput(self):
        past_bandwidths = self.mem_past_throughput[-5:]
        while past_bandwidths[0] == 0.0:
            past_bandwidths = past_bandwidths[1:]  # trick to avoid 0 througput past info (for initial states)
        # if ( len(state) < 5 ):
        #    past_bandwidths = state[3,-len(state):]
        # else:
        #    past_bandwidths = state[3,-5:]
        bandwidth_sum = 0
        for past_val in past_bandwidths:
            bandwidth_sum += (1 / float(past_val))
        future_bandwidth = 1.0 / (bandwidth_sum / len(past_bandwidths))
        return future_bandwidth

    def quality_from_throughput(self, tput):  # should be in kbit/s but mine is in Byte/sec
        p = self.segment_length_t  # chunk length in ms

        quality = 0
        while (quality + 1 < len(self.VIDEO_BIT_RATE_T) and p * self.VIDEO_BIT_RATE_T[quality + 1] / tput <= p):  # mi sta dicendo che se per scaricare
            # un chunk ci metto meno che renderizzarlo allora alzo la qualità finche questo non è piu vero, in tutto questo c era anche un parametro di delay che
            # in park non mi viene restituito poiche le traccie non hanno la latenza o delay, non è chiaro#
            quality += 1
        return quality

    def get_quality_delay(self):
        throughput = self.estimated_throughput()
        quality = self.quality_from_throughput(throughput * self.safety_factor)
        #get_buffer_level() = self.last_state[2]

        if not self.no_ibr:
            # insufficient buffer rule
            safe_size = self.ibr_safety * (self.last_state[2])*throughput #- latency) * throughput
            self.ibr_safety *= self.low_buffer_safety_factor_init
            self.ibr_safety = max(self.ibr_safety, self.low_buffer_safety_factor)
            for q in range(quality):
                if self.VIDEO_BIT_RATE_T[q + 1] * self.segment_length_t > safe_size:
                    quality = q
                    break

        return quality

    def do_action(self):
        #return self.quality_from_buffer()
        if not self.first_action:
            return self.get_quality_delay()
        else:
            self.first_action=False
            return self.DEFAULT_QUALITY

    def reset_state(self):
        self.last_state = None
        self.last_obs = None
        self.first_action = True
        self.mem_past_throughput = []

    def obs_to_state(self):
        self.last_state=self.last_obs
        self.last_state[2] =self.last_state[2]*1000  # buffer in millissec
        self.last_state[0] = self.last_state[0]*0.008 # in kbit/s
        self.mem_past_throughput.append(self.last_state[0])  # in kbits

class BB_agent(Agent):
    # Bola original uses kbps
    VIDEO_BIT_RATE_BB = [i * 1000 for i in VIDEO_BIT_RATE]
    reservoir = 5
    cushion = 10

    def __init__(self, state_space, action_space):
        Agent.__init__(self, state_space, action_space)
        self.state_space = state_space
        self.action_space = action_space
        self.last_action = None
        self.last_obs = None
        self.last_state = None
        self.last_info = None
        self.bitrateArray=self.VIDEO_BIT_RATE_BB
        self.first_action=True

    def getBitrateBB(self):
        tmpBitrate = 0
        tmpQuality = 0
        bLevel = self.last_state[2]
        if (bLevel <= self.reservoir):
            tmpBitrate = self.bitrateArray[0]
        elif (bLevel > self.reservoir + self.cushion):
            tmpBitrate = self.bitrateArray[12]
        else:
            tmpBitrate = self.bitrateArray[0] + (self.bitrateArray[12] - self.bitrateArray[0]) * (bLevel - self.reservoir) / self.cushion


        # findout matching quality level
        for i in range(12,-1,-1):
            if (tmpBitrate >= self.bitrateArray[i]):
                tmpQuality = i
                break

            tmpQuality = i

        # return 9;

        return tmpQuality
        # return 0;

    def do_action(self):
        #return self.quality_from_buffer()
        if not self.first_action:
            return self.getBitrateBB()
        else:
            self.first_action=False
            self.last_state=[0,0,0]
            return self.getBitrateBB()

    def reset_state(self):
        self.last_state = None
        self.last_obs = None
        self.first_action = True

    def obs_to_state(self):
        self.last_state=self.last_obs

class HYB_agent(Agent):
    # Bola original uses kbps
    beta = 0.25


    def __init__(self, state_space, action_space):
        Agent.__init__(self, state_space, action_space)
        self.state_space = state_space
        self.action_space = action_space
        self.last_action = None
        self.last_obs = None
        self.last_state = None
        self.last_info = None
        self.first_action = True
        self.mem_past_throughput = []

    def estimated_throughput(self):
        past_bandwidths = self.mem_past_throughput[-5:]
        while past_bandwidths[0] == 0.0:
            past_bandwidths = past_bandwidths[1:]  # trick to avoid 0 througput past info (for initial states)
        # if ( len(state) < 5 ):
        #    past_bandwidths = state[3,-len(state):]
        # else:
        #    past_bandwidths = state[3,-5:]
        bandwidth_sum = 0
        for past_val in past_bandwidths:
            bandwidth_sum += (1 / float(past_val))
        future_bandwidth = 1.0 / (bandwidth_sum / len(past_bandwidths))
        return future_bandwidth

    def getBitrateHYB(self):
        if not self.first_action:
            quality = 0
            B = self.estimated_throughput() #bytes/sec
            L = self.last_state[2] #sec
            for i in range(5,-1,-1):
                size = self.last_state[i+5] #next chunk size decreasing
                if size/B < L*self.beta:
                    quality = i
                    break
            return quality
        else:
            self.first_action = False
            return self.DEFAULT_QUALITY


    def do_action(self):
        # return self.quality_from_buffer()
        return self.getBitrateHYB()

    def reset_state(self):
        self.last_state = None
        self.last_obs = None
        self.first_action = True
        self.mem_past_throughput = []

    def obs_to_state(self):
        self.last_state = self.last_obs
        self.mem_past_throughput.append(self.last_state[0]) #bytes/s














