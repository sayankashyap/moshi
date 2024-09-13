import moshiProcessorUrl from "../../audio-processor.ts?worker&url";
import { FC, useEffect, useMemo, useState, useCallback, useRef, MutableRefObject } from "react";
import { getAPIClient } from "./api/client";
import eruda from "eruda";
import { useSearchParams } from "react-router-dom";
import { Conversation } from "../Conversation/Conversation";
import { Button } from "../../components/Button/Button";
import { useModelParams } from "../Conversation/hooks/useModelParams";
import { ModelParams } from "../Conversation/components/ModelParams/ModelParams";
import { env } from "../../env";

type Status = "connecting" | "in_queue" | "has_credentials" | "error" | "no_queue" | "idle"| "bypass";


export const Queue:FC = () => {
  const [searchParams] = useSearchParams();
  let queueId = searchParams.get("queue_id");
  if (!queueId) {
    queueId = 'talktomoshi';
  }
  const workerAddr = searchParams.get("worker_addr");
  const [sessionId, setSessionId] = useState<number|null>(null);
  const [sessionAuthId, setSessionAuthId] = useState<string|null>(null);
  const [workerAuthId, setWorkerAuthId] = useState<string|null>(null);
  const [currentPosition, setCurrentPosition] = useState<string|null>(null);
  const [error, setError] = useState<string|null>(null);
  const [shouldConnect, setShouldConnect] = useState<boolean>(false);
  const [hasMicrophoneAccess, setHasMicrophoneAccess] = useState<boolean>(false);
  const [showMicrophoneAccessMessage, setShowMicrophoneAccessMessage] = useState<boolean>(false);
  const modelParams = useModelParams();
  const modalRef = useRef<HTMLDialogElement>(null);

  const currentUrl = new URL(window.location.href);
  const hostname = currentUrl.hostname;
  const searchParamsForGeo = currentUrl.search;

  const constructNewGeoUrl = () => {
    let newHostname = '';
    let currentGeo = '';
    if (hostname.endsWith('us.moshi.chat')) {
      currentGeo = 'US';
      newHostname = hostname.replace('us.moshi.chat', 'moshi.chat');
    } else if (hostname.endsWith('eu.moshi.chat')) {
      currentGeo = 'EU';
      newHostname = hostname.replace('eu.moshi.chat', 'us.moshi.chat');
    } else if (hostname.endsWith('moshi.chat')) {
      currentGeo = 'EU';
      newHostname = hostname.replace('moshi.chat', 'us.moshi.chat');
    }
    const otherGeo = currentGeo === 'EU' ? 'US' : 'EU';
    const port = currentUrl.port;
    const newUrl = `${currentUrl.protocol}//${newHostname}:${port}${currentUrl.pathname}${searchParamsForGeo}`;
    return [newUrl, currentGeo, otherGeo];
  };
  const [newGeoUrl, currentGeo, otherGeo] = constructNewGeoUrl();

  const audioContext = useRef<AudioContext | null>(null);
  const worklet = useRef<AudioWorkletNode | null>(null);
  // enable eruda in development
  useEffect(() => {
    if(env.VITE_ENV === "development") {
      eruda.init();
    }
    () => {
      if(env.VITE_ENV === "development") {
        eruda.destroy();
      }
    };
  }, []);

  const getMicrophoneAccess = useCallback(async () => {
    try {
      await window.navigator.mediaDevices.getUserMedia({ audio: true });
      setHasMicrophoneAccess(true);
      return true;
    } catch(e) {
      console.error(e);
      setShowMicrophoneAccessMessage(true);
      setHasMicrophoneAccess(false);
    }
    return false;
}, [setHasMicrophoneAccess, setShowMicrophoneAccessMessage, setShouldConnect]);

  const startProcessor = useCallback(async () => {
    if(!audioContext.current) {
      audioContext.current = new AudioContext();
    }
    if(worklet.current) {
      return;
    }
    let ctx = audioContext.current;
    ctx.resume();
    try {
      worklet.current = new AudioWorkletNode(ctx, 'moshi-processor');
    } catch (err) {
      await ctx.audioWorklet.addModule(moshiProcessorUrl);
      worklet.current = new AudioWorkletNode(ctx, 'moshi-processor');
    }
    worklet.current.connect(ctx.destination);
  }, [audioContext, worklet]);

  const onConnect = useCallback(async() => {
      await startProcessor();
      const hasAccess = await getMicrophoneAccess();
      if(hasAccess) {
        setShouldConnect(true);
      }
  }, [setShouldConnect, startProcessor, getMicrophoneAccess]);

  const client = useMemo(() => {
    return getAPIClient(env.VITE_QUEUE_API_PATH)
  }, [env.VITE_QUEUE_API_PATH]);

  if(hasMicrophoneAccess && audioContext.current && worklet.current) {
    return (
      <Conversation
        workerAddr={workerAddr ?? ""}
        audioContext={audioContext as MutableRefObject<AudioContext>}
        worklet={worklet as MutableRefObject<AudioWorkletNode>}
        {...modelParams}
      />
    );
  }

  return (
    <div className="text-white text-center h-screen w-screen p-4 flex flex-col items-center ">
      <div>
        <h1 className="text-4xl">Moshi</h1>
        {/*
          To add more space to the top add padding to the top of the following div
          by changing the pt-4 class to pt-8 or pt-12. (see: https://tailwindcss.com/docs/padding)
          If you'd like to move this part to the bottom of the screen, change the class to pb-4 or pb-8 and move the following so it is contained by the last one in the page.
          Font size can be changed by changing the text-sm class to text-lg or text-xl. (see : https://tailwindcss.com/docs/font-size)
          As for the links you can use the one below as an example and add more by copying it and changing the href and text.
        */}
        <div className="pt-8 text-sm flex justify-center items-center flex-col ">
          <div className="presentation text-left">
          <p><span className='cute-words'>Moshi</span> is an experimental conversational AI. </p>
          <p>Take everything it says with a grain of <span className='cute-words'>salt</span>.</p>
          <p>Conversations are limited to <span className='cute-words'>5 min</span>.</p>
          <p>Moshi <span className='cute-words'>thinks</span> and <span className='cute-words'>speaks</span> at the same time.</p>
          <p>Moshi can <span className='cute-words'>listen</span> and <span className='cute-words'>talk</span> at all time: <br/>maximum flow between you and <span className='cute-words'>Moshi</span>.</p>
          <p>Ask it to do some <span className='cute-words'>Pirate</span> role play, how to make <span className='cute-words'>Lasagna</span>,
            or what <span className='cute-words'>movie</span> it watched last.</p>
          <p>We strive to support all browsers, Chrome works best.</p>
          <p>Baked with &lt;3 @<a href="https://kyutai.org/" className='cute-words underline'>Kyutai</a>.</p>
          { currentGeo !== '' && <p>You are on the <span className='cute-words'>{currentGeo}</span> demo.
            Depending on your location, maybe the <a href={newGeoUrl} className="cute-words underline">{otherGeo} demo</a> will offer better latency.</p>}
          </div>
        </div>
      </div>
      <div className="flex flex-grow justify-center items-center flex-col">
      {!workerAddr && <p className="text-center text-red-800 text-2xl">No worker_addr provided in the url, add one!</p>}
      {(
        <>
          {showMicrophoneAccessMessage &&
            <p className="text-center">Please enable your microphone before proceeding</p>
          }
          <Button onClick={async () => await onConnect()}>Connect</Button>
          <Button className="absolute top-4 right-4" onClick={()=> modalRef.current?.showModal()}>Settings</Button>
            <dialog ref={modalRef} className="modal">
              <div className="modal-box border-2 border-white rounded-none flex justify-center bg-black">
                <ModelParams {...modelParams} isConnected={shouldConnect} modal={modalRef}/>
              </div>
              <form method="dialog" className="modal-backdrop">
                <button>Close</button>
              </form>
            </dialog>
        </>
      )}
      </div>
      <div className="text-center flex justify-end items-center flex-col">
        <a target="_blank" href="https://kyutai.org/moshi-terms.pdf" className="text-center">Terms of Use</a>
        <a target="_blank" href="https://kyutai.org/moshi-privacy.pdf" className="text-center">Privacy Policy</a>
      </div>
    </div>
  )
};
